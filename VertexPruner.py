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
import torch.nn.functional as F
from training_utils import LinePlot

kl_loss = torch.nn.KLDivLoss(reduction="none")

class VertexPruner(torch.nn.Module):
    def __init__(self, 
                 model, 
                 pruning_cfg, 
                 init_modes,
                 mask_sampler, 

                #  False value not supported
                 parallel_inference=True, 
                 inference_mode=False, 
                 cache_compressed_attn=True, 
                 ablation_backward=False
                 ):
        super().__init__()
        self.base_model = model
        self.pruning_cfg = pruning_cfg
        self.mask_sampler = mask_sampler

        init_modes_attention, init_modes_mlp = init_modes
        self.modal_attention = torch.nn.Parameter(init_modes_attention)
        self.modal_mlp = torch.nn.Parameter(init_modes_mlp)
        self.node_reg = 0
        # self.cache_compressed_attn = cache_compressed_attn
        self.inference_mode = inference_mode
        self.parallel_inference = parallel_inference
        self.ablation_backward = ablation_backward
        self.disable_hooks = False
        
        columns =  ['kl_loss', 'complexity_loss', 'temp', 'temp_cond', 'temp_count', 'temp_reg']
        self.log = LinePlot(columns)

        # self.cache_hooks = self.get_cache_hooks()
        self.patching_hooks = self.get_patching_hooks()

        self.last_token_mask = None
        
    def set_log(self, log):
        self.log = log
    
    def add_patching_hooks(self):
        for name, hook in self.patching_hooks:
            self.base_model.add_hook(name, hook)

    # attentions: (batch_size + batch_size * n_samples) x seq_len x n_heads x d_model
    # constants: n_heads x d_model
    # prune mask: (batch_size * n_samples) x n_heads, 0 = prune, 1 = keep
    def pruning_hook_attention_all_tokens(self, layer_no, attentions, hook):
        bsz = self.pruning_cfg.batch_size

        prune_mask = self.mask_sampler.sampled_mask['attn'][layer_no].unsqueeze(1).unsqueeze(-1)
        attentions[bsz:] = (1-prune_mask) * self.modal_attention[layer_no] + prune_mask * attentions[bsz:].clone()

        # prune_idx = prune_mask.clone()
        # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
        return attentions
    
   # attentions: (batch_size + batch_size * n_samples) x seq_len x n_heads x d_model
    # constants: n_heads x d_model
    # prune mask: (batch_size * n_samples) x n_heads, 0 = prune, 1 = keep
    def pruning_hook_mlp_all_tokens(self, layer_no, mlp_out, hook):
        bsz = self.pruning_cfg.batch_size

        prune_mask = self.mask_sampler.sampled_mask['mlp'][layer_no].unsqueeze(1).unsqueeze(-1)
        mlp_out[bsz:] = (1-prune_mask) * self.modal_mlp[layer_no] + prune_mask * mlp_out[bsz:].clone()

        # prune_idx = prune_mask.clone()
        # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
        return mlp_out

    def final_hook_last_token(self, out, hook):
        if (self.inference_mode and not self.parallel_inference) or self.disable_hooks:
            out = out.unsqueeze(0)
        else:
            out = out.unflatten(0, (-1, self.pruning_cfg.batch_size))
        out = (out * self.last_token_mask.unsqueeze(-1)).sum(dim=2)
        return out

    def get_patching_hooks(self, ):
        attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"
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
        return self.modal_attention
    
    def complexity_loss(self, sampling_params):
        return (sampling_params[...,0]-sampling_params[...,1].relu() * (math.log(-self.pruning_cfg.hard_concrete_endpoints[0]/self.pruning_cfg.hard_concrete_endpoints[1]))).sigmoid()  
                
    def forward(self, batch, last_token_pos, graph_suffix=None, complexity_mean=False, return_output=False, timing=True):
        if timing:
            end = []
            for x in range(6):
                end.append(torch.cuda.Event(enable_timing=True))
            end[0].record()

        all_sampling_params = self.mask_sampler()

        with torch.no_grad():
            last_token_mask = torch.zeros_like(batch).to(self.pruning_cfg.device)
            last_token_mask[torch.arange(last_token_mask.shape[0]), last_token_pos] = 1
        
        self.last_token_mask = last_token_mask

        if timing:
            end[1].record()
        
        n_samples = 1 if self.inference_mode else self.pruning_cfg.n_samples
        pruned_output = self.base_model(
            batch.repeat(n_samples+1,1)
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
        # io_loss = target_results - ablated_results

        if self.inference_mode:
            return kl_losses

        # alphas already logged
        complexity_loss = self.complexity_loss(all_sampling_params)
                    
        temperature_loss = all_sampling_params[...,1].square()

        temp_c = self.pruning_cfg.temp_scheduler(self.log)

        loss = kl_losses.mean() + self.pruning_cfg.lamb * complexity_loss.sum() + temp_c * temperature_loss.sum()

        # end[4].record()

        with torch.no_grad():
            avg_temp = all_sampling_params[...,1].relu().mean().item()
            temp_cond = torch.nan_to_num((all_sampling_params[...,1]-1).relu().sum() / (all_sampling_params[...,1] > 1).sum(), nan=0, posinf=0, neginf=0).item() + 1
            temp_count = (2*all_sampling_params[:,1].relu().sigmoid()-1).mean().item()

            log_entry = {
                "kl_loss": kl_losses.mean().item(), 
                "complexity_loss": complexity_loss.mean().item() if complexity_mean else complexity_loss.sum().item(),
                "temp": avg_temp,
                "temp_cond": temp_cond,
                "temp_count": temp_count,
                "temp_reg": temp_c
            }

            self.log.add_entry(log_entry)

            if graph_suffix is not None:
                j = graph_suffix
                sns.histplot(kl_losses.flatten().detach().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/kl-loss{j}.png")
                plt.close()

                sns.histplot(torch.cat([ts.flatten() for k in self.mask_sampler.sampled_mask for ts in self.mask_sampler.sampled_mask[k]], dim=0).detach().flatten().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/mask{j}.png")
                plt.close()

                sns.histplot(x=all_sampling_params[:,0].sigmoid().detach().flatten().cpu(), y=all_sampling_params[:,1].detach().flatten().cpu(), bins=100)
                plt.savefig(f"{self.pruning_cfg.folder}/params-probs{j}.png")
                plt.close()

                sns.histplot(x=all_sampling_params[:,0].detach().flatten().cpu(), y=all_sampling_params[:,1].detach().flatten().cpu(), bins=100)
                plt.savefig(f"{self.pruning_cfg.folder}/params-logits{j}.png")
                plt.close()

            print("KL:", kl_losses.mean().item())
            print("Complexity:", complexity_loss.sum().item(), "out of", complexity_loss.nelement())
            print("Avg temperature", avg_temp)
            print("Avg temp > 1", temp_cond)
            print("Temp count", temp_count)

        return loss, all_sampling_params