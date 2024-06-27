import torch
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
import math
from functools import partial
import torch.optim
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import LinePlot

kl_loss = torch.nn.KLDivLoss(reduction="none")

class Pruner(torch.nn.Module):
    def __init__(self, 
                 model, 
                 pruning_cfg, 
                 mask_sampler, 

                 # run the clean and ablated runs at the same time
                 parallel_inference=False, 

                 # ABLATION TYPE PARAMS
                 # do we pass in counterfactual inputs to do patching?
                 counterfactual_mode=False, 
                 # does the patched value depend on sequence position?
                 condition_pos=False,
                 # value for constant ablations
                 init_modes=None,
                 ):
        
        super().__init__()
        self.base_model = model
        self.pruning_cfg = pruning_cfg
        self.mask_sampler = mask_sampler

        self.parallel_inference = parallel_inference
        self.disable_hooks = False

        columns = ['kl_loss', *self.mask_sampler.log_columns]
        self.log = LinePlot(columns)

        self.patching_hooks = self.get_patching_hooks()

        self.last_token_mask = None

        self.counterfactual_mode = counterfactual_mode
        self.condition_pos = condition_pos
        if self.counterfactual_mode:
            self.modal_attention = None 
            self.modal_mlp = None
            self.perms = None
            columns =  ['kl_loss', *self.mask_sampler.log_columns]
            self.log = LinePlot(columns)
        else:
            self.reset_parameters(init_modes)

        self.pause_log = False
        self.seq_len = None

    def reset_parameters(self, init_modes):
        init_modes_attention, init_modes_mlp = init_modes
        self.modal_attention = torch.nn.Parameter(init_modes_attention.clone())
        self.modal_mlp = torch.nn.Parameter(init_modes_mlp.clone())
        self.null_vals = {'attn': self.modal_attention, 'mlp': self.modal_mlp}
        columns =  ['kl_loss', *self.mask_sampler.log_columns]
        self.log = LinePlot(columns)

    def set_log(self, log):
        self.log = log

    def add_cache_hooks(self):
        for name, hook in self.cache_hooks.values():
            self.base_model.add_hook(name, hook)
    
    def add_patching_hooks(self):
        for name, hook in self.patching_hooks.values():
            self.base_model.add_hook(name, hook)

    def time_hook(self, name, f, *args, **kwargs):
        end = []
        for i in range(2):
            end.append(torch.cuda.Event(enable_timing=True))
        end[0].record()

        x = f(*args, **kwargs)
        end[1].record()
        torch.cuda.synchronize()

        print(name, end[0].elapsed_time(end[1]))

        return x

    def early_term(self, decline_pct=.03):
        if self.log.t < 500:
            return 0
        
        kl_loss_decl, _ = self.log.stat_sig_growth("kl_loss")
        complex_loss_decl, _ = self.log.stat_sig_growth("complexity_loss")
        temp = self.log.stat_book["temp"][-1]

        if kl_loss_decl < 0.01 and complex_loss_decl < decline_pct and temp < 1e-2:
            self.log.early_term_count += 1
        else:
            self.log.early_term_count = max(0, self.log.early_term_count - 2)
        return self.log.early_term_count

    def get_modes(self):
        return torch.cat([self.modal_attention.flatten(start_dim=1,end_dim=2), self.modal_mlp], dim=0)

    def setup_inference(self, batch, last_token_pos):
        if self.condition_pos:
            # seq_pos x i x n_heads x d_head
            null_attn = self.modal_attention

            # seq_pos x i x d_model
            null_mlp = self.modal_mlp

            assert null_mlp.shape[0] == null_attn.shape[0]
            diff = batch.shape[1] - null_mlp.shape[0]
            if diff <= 0:
                null_attn = null_attn[:batch.shape[1]]
                null_mlp = null_mlp[:batch.shape[1]]
            else:
                null_attn = torch.cat([null_attn, null_attn[[-1]].expand(diff,-1,-1,-1)], dim=0)
                null_mlp = torch.cat([null_mlp, null_mlp[[-1]].expand(diff,-1,-1)], dim=0)
            
            self.null_vals['attn'] = null_attn
            self.null_vals['mlp'] = null_mlp
            
        elif self.counterfactual_mode:
            self.perms = [torch.randperm(n).to(batch.device) for n in last_token_pos]

        with torch.no_grad():
            last_token_mask = torch.zeros_like(batch).to(batch.device)
            last_token_mask[torch.arange(last_token_mask.shape[0]), last_token_pos] = 1
        self.last_token_mask = last_token_mask
        self.seq_len = batch.shape[1]
    
    # graph_suffix: current time, pass if we want to plot a histogram of KL loss, mask params
    # return output: just return the model output
    # separate loss: separate KL and mask loss
    def forward(self, batch, last_token_pos, counterfactual=None, graph_suffix=None, return_output=False, timing=True, print_loss=True, separate_loss=False):
        if timing:
            end = []
            for x in range(6):
                end.append(torch.cuda.Event(enable_timing=True))
            end[0].record()
        
        self.setup_inference(batch, last_token_pos)

        n_samples = self.pruning_cfg.n_samples
        
        if self.mask_sampler.use_temperature:
            self.mask_sampler.set_temp_c(self.pruning_cfg.temp_scheduler(self.log))
        mask_loss, mask_details = self.mask_sampler()

        if timing:
            end[1].record()
        
        # CONVENTION: since we repeat the batch tokens n_samples times, the correct unflattened shape for embeddings is [n_samples, batch_size, seq, d_model]

        model_input = batch.repeat(n_samples+(1 if self.parallel_inference else 0),1)
        if self.counterfactual_mode:
            if counterfactual is None:
                raise Exception("Expected counterfactual")
            model_input = torch.cat([counterfactual, model_input], dim=0)
        
        pruned_output = self.base_model(
            model_input
        ).log_softmax(dim=-1)

        if timing:
            end[2].record()

        if return_output:
            if timing: 
                torch.cuda.synchronize()
                print("Cuda time", end[1].elapsed_time(end[2]))
            return pruned_output
        
        if self.parallel_inference:
            orig_output = pruned_output[0]
            pruned_output = pruned_output[1:]
        else:
            self.disable_hooks = True
            with torch.no_grad():
                orig_output = self.base_model(batch)
                orig_output = orig_output.log_softmax(dim=-1)
            self.disable_hooks = False
        
        if timing:
            end[3].record()
            torch.cuda.synchronize()
            for i in range(1,4):
                print("Cuda time", end[i-1].elapsed_time(end[i]))

        kl_losses = kl_loss(pruned_output, orig_output.exp()).sum(dim=-1)
        loss = kl_losses.mean() + mask_loss

        with torch.no_grad():
            log_entry = {
                "kl_loss": kl_losses.mean().item(), 
                **mask_details
            }
            if self.pause_log is False:
                self.log.add_entry(log_entry)

            if graph_suffix is not None:
                j = graph_suffix
                sns.histplot(kl_losses.flatten().detach().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/kl-loss{j}.png")
                plt.close()

                self.mask_sampler.record_state(j)

            if print_loss:
                print("KL:", kl_losses.mean().item())
        
        if separate_loss:
            return kl_losses, mask_loss
        else:
            return loss