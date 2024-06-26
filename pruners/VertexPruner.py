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

class VertexPruner(torch.nn.Module):
    def __init__(self, 
                 model, 
                 pruning_cfg, 
                 mask_sampler, 

                #  False value not supported
                 parallel_inference=True, 
                 ablation_backward=False,

                 # some ablation types need to run cache
                 counterfactual_mode=False, 
                 condition_pos=False,
                 init_modes=None,
                 ):
        super().__init__()
        self.base_model = model
        self.pruning_cfg = pruning_cfg
        self.mask_sampler = mask_sampler
        self.all_gradients = True

        self.parallel_inference = parallel_inference
        self.ablation_backward = ablation_backward
        self.disable_hooks = False

        columns = ['kl_loss', *self.mask_sampler.log_columns]
        self.log = LinePlot(columns)

        self.patching_hooks = self.get_patching_hooks()

        self.last_token_mask = None
        self.seq_len = None

        self.counterfactual_mode = counterfactual_mode
        self.condition_pos = condition_pos
        if self.counterfactual_mode:
            self.perms = None
        else:
            init_modes_attention, init_modes_mlp = init_modes
            self.modal_attention = torch.nn.Parameter(init_modes_attention)
            self.modal_mlp = torch.nn.Parameter(init_modes_mlp)

        
    def set_log(self, log):
        self.log = log
    
    def add_patching_hooks(self):
        for name, hook in self.patching_hooks:
            self.base_model.add_hook(name, hook)
    
    def process_null_val(self, node_type, layer_no):
        if node_type == "attn":
            null_val = self.modal_attention[...,layer_no,:,:]
        elif node_type == "mlp":
            null_val = self.modal_mlp[...,layer_no,:]
        else:
            raise Exception("vertex type")
        
        if self.condition_pos:
            # seq_pos x i x n_heads x d_head
            diff = self.seq_len - null_val.shape[0]
            if diff <= 0:
                null_val = null_val[:self.seq_len]
            else:
                null_val = torch.cat([null_val, null_val[[-1]].expand(diff, *[-1 for _ in null_val.shape[1:]])], dim=0)
        
        return null_val

    # attentions: (batch_size + batch_size * n_samples) x seq_len x n_heads x d_model
    # constants: n_heads x d_head
    # prune mask: (batch_size * n_samples) x n_heads, 0 = prune, 1 = keep
    def pruning_hook_attention_all_tokens(self, layer_no, attentions, hook):
        bsz = self.pruning_cfg.batch_size
        if self.counterfactual_mode:
            # first batch_size are counterfactual, then next batch_size are true
            null_val = attentions[:bsz]
            bsz = 2 * bsz

            if not self.condition_pos:
                for i, p in enumerate(self.perms):
                    null_val[i,:p.shape[0]] = null_val[i,p]

            null_val = null_val.repeat(self.pruning_cfg.n_samples, 1, 1, 1)
        else:
            null_val = self.process_null_val("attn", layer_no)
        
        try:
            bos_out = attentions[:,[0]].clone().detach()
            prune_mask = self.mask_sampler.sampled_mask['attn'][layer_no].unsqueeze(1).unsqueeze(-1)
            ablated_attn = (
                (prune_mask < 0.001) * (1-prune_mask) * null_val
                + (prune_mask >= 0.001) * (1-prune_mask) * null_val.detach()
            ) + prune_mask * attentions[bsz:].clone()

            if self.all_gradients:
                attentions[bsz:] = ablated_attn
            else:
                # for grad sampling (attribution patching): we are adding mask_perturb, drop gradient if it rounds to 1
                attentions[bsz:] = (
                    (prune_mask <= 1 - 1e-3) * ablated_attn
                    + (prune_mask > 1 - 1e-3) * prune_mask.detach() * attentions[bsz:].clone()
                )
        except Exception as e:
            print(bsz)
            print(null_val.shape)
            print(attentions.shape)
            print(prune_mask.shape)
            raise e

        # prune_idx = prune_mask.clone()
        # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
        # return attentions
        attentions[:,[0]] = bos_out
        return attentions

    # attentions: (batch_size + batch_size * n_samples) x seq_len x d_model
    # constants: d_model
    # prune mask: (batch_size * n_samples), 0 = prune, 1 = keep
    def pruning_hook_mlp_all_tokens(self, layer_no, mlp_out, hook):
        bsz = self.pruning_cfg.batch_size

        if self.counterfactual_mode:
            # first batch_size are counterfactual, then next batch_size are true
            null_val = mlp_out[:bsz]
            bsz = 2 * bsz

            if not self.condition_pos:
                for i, p in enumerate(self.perms):
                    null_val[i,:p.shape[0]] = null_val[i,p]

            null_val = null_val.repeat(self.pruning_cfg.n_samples, 1, 1)
        else:
            null_val = self.process_null_val("mlp", layer_no)

        try:
            bos_out = mlp_out[:,[0]].clone().detach()
            prune_mask = self.mask_sampler.sampled_mask['mlp'][layer_no].unsqueeze(1).unsqueeze(-1)
            ablated_mlp = (
                (prune_mask < 0.001) * (1-prune_mask) * null_val
                + (prune_mask >= 0.001) * (1-prune_mask) * null_val.detach()
            ) + prune_mask * mlp_out[bsz:].clone()

            if self.all_gradients:
                mlp_out[bsz:] = ablated_mlp
            else:
                # for grad sampling (attribution patching): we are adding mask_perturb, drop gradient if it rounds to 1
                mlp_out[bsz:] = (
                    (prune_mask <= 1 - 1e-3) * ablated_mlp
                    + (prune_mask > 1 - 1e-3) * prune_mask.detach() * mlp_out[bsz:].clone()
                )

            # prune_idx = prune_mask.clone()
            # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]

            # return mlp_out
        except Exception as e:
            print(mlp_out.shape)
            print(prune_mask.shape)
            print(null_val.shape)
            raise e
        
        mlp_out[:,[0]] = bos_out
        return mlp_out

    def final_hook_last_token(self, out, hook):
        bsz = self.pruning_cfg.batch_size

        # remove counterfactuals
        if self.counterfactual_mode:
            out = out[bsz:]

        if self.disable_hooks:
            out = out.unsqueeze(0)
        else:
            out = out.unflatten(0, (-1, bsz))
        out = (out * self.last_token_mask.unsqueeze(-1)).sum(dim=2)
        return out

    def get_patching_hooks(self):
        # attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"
        attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_z"
        mlp_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"
        final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

        n_layers = self.base_model.cfg.n_layers
        
        return [
                *[(partial(attention_points_filter, layer_no), 
                   partial(self.pruning_hook_attention_all_tokens, layer_no)
                ) for layer_no in range(n_layers)],
                *[(partial(mlp_out_filter, layer_no), 
                   partial(self.pruning_hook_mlp_all_tokens, layer_no)
                ) for layer_no in range(n_layers)],
                (final_embed_filter, self.final_hook_last_token)
            ]
    
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

    # graph_suffix: current time, pass if we want to plot a histogram of KL loss, mask params
    def forward(self, batch, last_token_pos, counterfactual=None, graph_suffix=None, return_output=False, timing=True, separate_loss=False):
        if timing:
            end = []
            for x in range(6):
                end.append(torch.cuda.Event(enable_timing=True))
            end[0].record()
        
        with torch.no_grad():
            last_token_mask = torch.zeros_like(batch).to(self.pruning_cfg.device)
            last_token_mask[torch.arange(last_token_mask.shape[0]), last_token_pos] = 1
        
        if self.counterfactual_mode and not self.condition_pos:
            self.perms = [torch.randperm(n).to(batch.device) for n in last_token_pos]

        self.seq_len = batch.shape[1]
        self.last_token_mask = last_token_mask        
        n_samples = self.pruning_cfg.n_samples
        
        if self.mask_sampler.use_temperature:
            self.mask_sampler.set_temp_c(self.pruning_cfg.temp_scheduler(self.log))
        mask_loss, mask_details = self.mask_sampler()
        
        if timing:
            end[1].record()

        model_input = batch.repeat(n_samples+1,1)
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
        
        orig_output = pruned_output[0]
        pruned_output = pruned_output[1:]
        
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
            self.log.add_entry(log_entry)

            if graph_suffix is not None:
                j = graph_suffix
                sns.histplot(kl_losses.flatten().detach().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/kl-loss{j}.png")
                plt.close()

                self.mask_sampler.record_state(j)

            print("KL:", kl_losses.mean().item())

        if separate_loss:
            return kl_losses, mask_loss
        else:
            return loss