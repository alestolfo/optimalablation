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
from training_utils import LinePlot

kl_loss = torch.nn.KLDivLoss(reduction="none")

class EdgePruner(torch.nn.Module):
    def __init__(self, model, pruning_cfg, mask_sampler, parallel_inference=False, inference_mode=False, cache_compressed_attn=True, ablation_backward=False):
        super().__init__()
        self.base_model = model
        self.pruning_cfg = pruning_cfg
        self.mask_sampler = mask_sampler
        

        init_modes_attention, init_modes_mlp = pruning_cfg.init_modes()
        self.modal_attention = torch.nn.Parameter(init_modes_attention)
        self.modal_mlp = torch.nn.Parameter(init_modes_mlp)
        self.attention_cache = []
        self.mlp_cache = []
        self.cache_compressed_attn = cache_compressed_attn
        self.inference_mode = inference_mode
        self.parallel_inference = parallel_inference
        self.ablation_backward = ablation_backward

        if not self.cache_compressed_attn:
            model.cfg.use_attn_result = True

        self.cache_hooks = self.get_cache_hooks()

        self.post_bias = torch.stack([self.base_model.blocks[layer_no].attn.b_O.clone().detach() for layer_no in range(self.base_model.cfg.n_layers)], dim=0)

        if cache_compressed_attn:
            self.W_O = torch.stack([self.base_model.blocks[layer_no].attn.W_O.clone().detach() for layer_no in range(self.base_model.cfg.n_layers)], dim=0)
        
        self.log = LinePlot(['kl_loss', 'complexity_loss', 'temp', 'temp_cond', 'temp_count', 'temp_reg'])
        
    def set_log(self, log):
        self.log = log

    def cache_hook_all_tokens(self, storage, activations, hook):

        if self.parallel_inference:
            bsz = self.pruning_config.batch_size
            storage.append(activations[bsz:])
            return activations[:bsz]
        else:
            storage.append(activations)
            return activations

    # attention_constants: list of all constants for attention for layers thus far
    # mlp_constants: list of all constants for embed+mlp layers thus far
    # attention_cache: contains all attentions stored thus far, list of attention outputs by later
    # mlp_cache: list of mlp outputs by layer
    def pruning_edge_attention_hook_all_tokens(self, W_O, prune_mask, attn_constants, mlp_constants, total_post_bias, orig_in, hook):
        def prepend_orig(out):
            if self.parallel_inference:
                return torch.cat([orig_in[:self.pruning_cfg.batch_size], out], dim=0)
            return out
        # i is the current layer (0-indexed, equal to the number of layers before this one)
        # orig_in: batch x seq_pos x d_model
        # prune_mask[0]: (bsz * n_samples) x n_heads (dest) x i x n_heads (source)
        # attention_constants: i x n_heads (source) x d_model
        # attention_cache: i * [(bsz * n_samples) x seq_pos x n_heads (source) x d_model]

        # mlp_constants: (i+1) x d_model
        # mlp_cache: (i+1) * [(bsz * n_samples) x seq_pos x d_model]

        # mlp_mask: (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x 1 (d_model)

        # return orig_in
        mlp_mask = prune_mask[1].unsqueeze(1).unsqueeze(-1)
        
        out = (mlp_mask * torch.stack(self.mlp_cache, dim=-2).unsqueeze(dim=2)).sum(dim=-2)
        # print((out-orig_in).square().sum())

        if mlp_constants is not None:
            out = out + ((1-mlp_mask) * mlp_constants).sum(dim=-2)

        if prune_mask[0] is None:
            return prepend_orig(out)
        
        # (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x n_heads (source) x 1 (d_model/d_head)
        attn_mask = prune_mask[0].unsqueeze(1).unsqueeze(-1)
        attn_term = attn_mask * torch.stack(self.attention_cache, dim=-3).unsqueeze(dim=2)

        # W_O: source_head x d_head x d_model
        if self.cache_compressed_attn:
            attn_term = einsum(
                        "batch pos dest_head prev_layer source_head d_head, \
                            prev_layer source_head d_head d_model -> \
                            batch pos dest_head d_model",
                        attn_term,
                        W_O
                )
        else:
            attn_term = attn_term.sum(dim=[-3,-2])
        out = out + attn_term + total_post_bias

        if mlp_constants is None:
            return prepend_orig(out + attn_constants)

        return prepend_orig(out + ((1-attn_mask) * attn_constants).sum(dim=[-3,-2]))

    # same as attentions except not parallelized
    # attention_constants: list of all constants for attention for layers thus far
    # mlp_constants: list of all constants for embed+mlp layers thus far
    # attention_cache: contains all attentions stored thus far, list of attention outputs by later
    # mlp_cache: list of mlp outputs by layer
    def pruning_edge_mlp_hook_all_tokens(self, W_O, prune_mask, attn_constants, mlp_constants, total_post_bias, orig_in, hook):     
        def prepend_orig(out):
            if self.parallel_inference:
                return torch.cat([orig_in[:self.pruning_cfg.batch_size], out], dim=0)
            return out
        # i is the current layer (0-indexed, equal to the number of layers before this one)
        # orig_in: batch x seq_pos x d_model
        # prune_mask[0]: (bsz * n_samples) x i x n_heads
        # attention_constants: i x n_heads x d_model
        # attention_cache: i * [(bsz * n_samples) x seq_pos x n_heads x d_model]

        # mlp_constants: (i+1) x d_model
        # mlp_cache: (i+1) * [(bsz * n_samples) x seq_pos x d_model]

        # (bsz * n_samples) x 1 (seq_pos) x i x 1 (d_model)
        mlp_mask = prune_mask[1].unsqueeze(1).unsqueeze(-1)

        out = (mlp_mask * torch.stack(self.mlp_cache, dim=2)).sum(dim=2)

        if attn_constants is not None:
            out = out + ((1-mlp_mask) * mlp_constants).sum(dim=2)

        # print((out - mlp_cache[0]).square().sum())

        if prune_mask[0] is None:
            return prepend_orig(out)
        
        # (bsz * n_samples) x 1 (seq_pos) x i x n_heads x 1 (d_model)
        attn_mask = prune_mask[0].unsqueeze(1).unsqueeze(-1)
        attn_term = attn_mask * torch.stack(self.attention_cache, dim=-3)

        # W_O: source_head x d_head x d_model
        if W_O is None: 
            attn_term = attn_term.sum(dim=[-3,-2])
        else:
            attn_term = einsum(
                        "batch pos prev_layer source_head d_head, \
                            prev_layer source_head d_head d_model -> \
                            batch pos d_model",
                        attn_term,
                        W_O
                )
        
        out = out + attn_term + total_post_bias

        if attn_constants is None:
            return prepend_orig(out + mlp_constants)
        
        return prepend_orig(out + ((1-attn_mask) * attn_constants).sum(dim=[2,3]))


    def pruning_edge_final_hook_all_tokens(self, inference_mode, last_token_mask, W_O, prune_mask, attn_constants, mlp_constants, total_post_bias, orig_in, hook):
        out = self.pruning_edge_mlp_hook_all_tokens(W_O, prune_mask, attn_constants, mlp_constants, total_post_bias, orig_in, hook)
        if not self.inference_mode:
            out = out.unflatten(0, (-1, self.pruning_cfg.batch_size))
        out = (out * last_token_mask.unsqueeze(-1)).sum(dim=2)
        return out

    def get_cache_hooks(self):
        embed_filter = lambda name: name == f"blocks.{0}.hook_resid_pre"
        attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"
        attention_compressed_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_z"
        mlp_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"

        n_layers = self.base_model.cfg.n_layers

        return [
            # cache embedding
            (embed_filter, 
            partial(self.cache_hook_all_tokens, self.mlp_cache)),

            # cache attention (at z if compressed)
            *[
                (partial(attention_compressed_filter if self.cache_compressed_attn else attention_points_filter, layer_no), 
                partial(self.cache_hook_all_tokens, self.attention_cache)) 
            for layer_no in range(n_layers)],

            # cache MLP
            *[
                (partial(mlp_points_filter, layer_no), 
                partial(self.cache_hook_all_tokens, self.mlp_cache)) 
            for layer_no in range(n_layers)],
        ]

    def get_pruning_hooks(self, prune_mask, last_token_mask, ablation_backward=False, layer_start=0):
        circs = ["q", "k", "v"]
        attention_in_filter = lambda layer_no, circ, name: name == f"blocks.{layer_no}.hook_{circ}_input"
        mlp_in_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_in"
        final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

        n_layers = self.base_model.cfg.n_layers
        
        return [
            # patch attention (recompute O-matrix if compressed)
            *[(partial(attention_in_filter, layer_no, circ), 
                partial(self.pruning_edge_attention_hook_all_tokens,
                        self.W_O[:layer_no] if self.cache_compressed_attn else None, 
                        [prune_mask["attn-attn"][layer_no][:,j] if layer_no > 0 else None, 
                            prune_mask["mlp-attn"][layer_no][:,j]], 
                        self.modal_attention[layer_no] if ablation_backward else self.modal_attention[:layer_no], 
                        None if ablation_backward else self.modal_mlp[:layer_no+1], 
                        self.post_bias[:layer_no].sum(dim=0))) 
            for layer_no in range(layer_start, n_layers) for j, circ in enumerate(circs)],

            # patch MLP (recompute O-matrix if compressed)
            *[(partial(mlp_in_filter, layer_no), 
                partial(self.pruning_edge_mlp_hook_all_tokens, 
                        self.W_O[:layer_no+1] if self.cache_compressed_attn else None, 
                        [prune_mask["attn-mlp"][layer_no], prune_mask["mlp-mlp"][layer_no]], 
                        None if ablation_backward else self.modal_attention[:layer_no+1], 
                        self.modal_mlp[layer_no] if ablation_backward else self.modal_mlp[:layer_no+1], 
                        self.post_bias[:layer_no+1].sum(dim=0))) 
            for layer_no in range(layer_start, n_layers)],

            # patch MLP (recompute O-matrix if compressed)
            (final_embed_filter, 
                partial(self.pruning_edge_final_hook_all_tokens, 
                        self.inference_mode,
                        last_token_mask,
                        self.W_O if self.cache_compressed_attn else None, 
                        [prune_mask["attn-mlp"][-1], prune_mask["mlp-mlp"][-1]], 
                        None if ablation_backward else self.modal_attention, 
                        self.modal_mlp[-1] if ablation_backward else self.modal_mlp, 
                        self.post_bias.sum(dim=0)))
        ]
    
    def early_term(self):
        if self.log.t < 500:
            return 0
        
        kl_loss_decl, _ = self.log.stat_sig_growth("kl_loss")
        complex_loss_decl, _ = self.log.stat_sig_growth("complexity_loss")
        temp = self.log.stat_book["temp"][-1]

        if kl_loss_decl < 0.01 and complex_loss_decl < 0.01 and temp < 1e-2:
            self.log.early_term_count += 1
        else:
            self.log.early_term_count = max(0, self.log.early_term_count - 2)
        return self.log.early_term_count

    def get_modes(self):
        return torch.cat([self.modal_attention.flatten(start_dim=0,end_dim=1), self.modal_mlp], dim=0)

    def forward(self, batch, last_token_pos, graph_suffix=None, layer_start=0, complexity_mean=False):

        end = []
        for x in range(6):
            end.append(torch.cuda.Event(enable_timing=True))
        end[0].record()

        prune_mask, all_sampling_params = self.mask_sampler()
        
        self.attention_cache.clear()
        self.mlp_cache.clear()

        with torch.no_grad():
            last_token_mask = torch.zeros_like(batch).to(self.pruning_cfg.device)
            last_token_mask[torch.arange(last_token_mask.shape[0]), last_token_pos] = 1
        
        end[1].record()

        pruned_output = self.base_model.run_with_hooks(
            batch.repeat(self.pruning_cfg.n_samples+(1 if self.parallel_inference else 0),1), 
            fwd_hooks=[
                *self.cache_hooks,
                *self.get_pruning_hooks(prune_mask, last_token_mask, layer_start=layer_start)
            ]).log_softmax(dim=-1)
        
        end[2].record()

        if self.parallel_inference:
            orig_output = pruned_output[0]
            pruned_output = pruned_output[1:]
        else:
            with torch.no_grad():
                orig_output = self.base_model(batch)
                orig_output = orig_output[torch.arange(orig_output.shape[0]), last_token_pos].log_softmax(dim=-1)
            
        end[3].record()
        torch.cuda.synchronize()

        for i in range(1,4):
            print("Cuda time", end[i-1].elapsed_time(end[i]))

        kl_losses = kl_loss(pruned_output, orig_output.exp()).sum(dim=-1)
        # io_loss = target_results - ablated_results

        if self.inference_mode:
            return kl_losses

        # alphas already logged
        complexity_loss = (all_sampling_params[:,0]-all_sampling_params[:,1].relu() * (math.log(-self.pruning_cfg.hard_concrete_endpoints[0]/self.pruning_cfg.hard_concrete_endpoints[1]))).sigmoid()
        temperature_loss = all_sampling_params[:,1].square()

        temp_c = self.pruning_cfg.temp_scheduler(self.log)

        loss = kl_losses.sum() + self.pruning_cfg.lamb * complexity_loss.mean() + temp_c * temperature_loss.mean()

        # end[4].record()

        with torch.no_grad():
            avg_temp = all_sampling_params[:,1].relu().mean().item()
            temp_cond = torch.nan_to_num((all_sampling_params[:,1]-1).relu().sum() / (all_sampling_params[:,1] > 1).sum(), nan=0, posinf=0, neginf=0).item() + 1
            temp_count = (2*all_sampling_params[:,1].relu().sigmoid()-1).mean().item()

            self.log.add_entry({
                "kl_loss": kl_losses.mean().item(), 
                "complexity_loss": complexity_loss.mean().item() if complexity_mean else complexity_loss.sum().item(),
                "temp": avg_temp,
                "temp_cond": temp_cond,
                "temp_count": temp_count,
                "temp_reg": temp_c
            })

            if graph_suffix is not None:
                j = graph_suffix
                sns.histplot(kl_losses.flatten().detach().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/kl-loss{j}.png")
                plt.close()

                sns.histplot(torch.cat([ts.flatten() for k in prune_mask for ts in prune_mask[k]], dim=0).detach().flatten().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/mask{j}.png")
                plt.close()

                sns.histplot(x=all_sampling_params[:,0].sigmoid().detach().flatten().cpu(), y=all_sampling_params[:,1].detach().flatten().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/params-probs{j}.png")
                plt.close()

                sns.histplot(x=all_sampling_params[:,0].detach().flatten().cpu(), y=all_sampling_params[:,1].detach().flatten().cpu())
                plt.savefig(f"{self.pruning_cfg.folder}/params-logits{j}.png")
                plt.close()

                self.log.plot(['kl_loss', 'complexity_loss'], save=f"{self.pruning_cfg.folder}/train-loss{j}.png")
                self.log.plot(['temp', 'temp_cond', 'temp_count', 'temp_reg'], save=f"{self.pruning_cfg.folder}/train-temp{j}.png")

            print("KL:", kl_losses.mean().item())
            print("Complexity:", complexity_loss.sum().item(), "out of", complexity_loss.shape[0])
            print("Avg temperature", avg_temp)
            print("Avg temp > 1", temp_cond)
            print("Temp count", temp_count)

        # end[5].record()

        # Waits for everything to finish running
        # torch.cuda.synchronize()

        # for i in range(1,4):
        #     print("Cuda time", end[i-1].elapsed_time(end[i]))

        return loss, all_sampling_params

class PruneMaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg):
        super().__init__()

        self.pruning_cfg = pruning_cfg

        self.sampling_params = torch.nn.ParameterDict({
            k: torch.nn.ParameterList([
                torch.nn.Parameter(p_init) for p_init in pruning_cfg.init_params[k]
            ]) for k in pruning_cfg.init_params
        })

        for param in self.parameters():
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))

    # beta and alpha should be same shape as x, or broadcastable
    # def f_concrete(x, beta, alpha):
    #     return ((x.log() - (1-x).log()) * beta - alpha.log()).sigmoid()

    def sample_prune_mask(self, unif, sampling_params):
        # back prop against log alpha
        endpts = self.pruning_cfg.hard_concrete_endpoints
        concrete = (((.001+unif).log() - (1-unif).log() + sampling_params[...,0])/(sampling_params[...,1].relu()+.001)).sigmoid()

        hard_concrete = ((concrete + endpts[0]) * (endpts[1] - endpts[0])).clamp(0,1)

        # n_layers x (total_samples = batch_size * n_samples) x n_heads
        return hard_concrete
        
    def fix_nans(self):
        fixed = True
        with torch.no_grad():
            sampling_params = self.get_sampling_params()
            
            nancount = sampling_params.isnan().sum()

            if nancount > 0:
                print("NANs", nancount)
                for k in self.sampling_params:
                    for ts in self.sampling_params[k]:
                        ts[ts[:,1].isnan().nonzero()[:,0],1] = 2/3
                        if ts.isnan().sum() > 0:
                            err = False
        return fixed

    def forward(self):
        pass

class PruneMaskJointSampler(PruneMaskSampler):
    def __init__(self, pruning_params):
        super().__init__(pruning_params)

    def get_sampling_params(self):
        return torch.cat([ts.flatten(start_dim=0, end_dim=-2) for k in self.sampling_params for ts in self.sampling_params[k]], dim=0)

    def forward(self):
        prune_mask = {}
        for k in self.sampling_params:
            prune_mask[k] = []
            for i in range(len(self.sampling_params[k])):
                # if sampling_params[k][i].nelement() == 0:
                #     prune_mask[k].append(None)
                #     continue
                unif = torch.rand((self.pruning_cfg.n_samples * self.pruning_cfg.batch_size, *self.sampling_params[k][i].shape[:-1])).to(self.pruning_cfg.device)
                prune_mask[k].append(self.sample_prune_mask(unif, self.sampling_params[k][i]))
        
        return prune_mask, self.get_sampling_params()
    
class PruneMaskIterativeSampler(PruneMaskSampler):
    def __init__(self, pruning_cfg, reverse=True):
        super().__init__(pruning_cfg)
        self.layers_to_prune = list(reversed(pruning_cfg.layers_to_prune)) if reverse else pruning_cfg.layers_to_prune
        self.layer_idx = -1
        self.prune_mask = self.pruning_cfg.constant_prune_mask
        self.next_layer()
        self.compute_edges()
    
    def next_layer(self):
        self.layer_idx += 1
        if self.layer_idx >= len(self.layers_to_prune):
            return False
        component_type, cur_layer = self.layers_to_prune[self.layer_idx]
        self.component_type = component_type
        self.cur_layer = cur_layer
        return True
    
    def get_sampling_params(self):
        return torch.cat([self.sampling_params[f"attn-{self.component_type}"][self.cur_layer].flatten(start_dim=0,end_dim=-2), self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer].flatten(start_dim=0,end_dim=-2)], dim=0)
    
    def compute_edges(self):
        self.total_edges = np.sum([(ts > 0).sum().item() for k in self.prune_mask for ts in self.prune_mask[k]])
    
    def freeze_params(self):
        self.prune_mask[f"attn-{self.component_type}"][self.cur_layer] = ((self.mask_sampler.sampling_params[f"attn-{self.component_type}"][self.cur_layer][...,0] > 0) * 1).unsqueeze(0)
        self.prune_mask[f"mlp-{self.component_type}"][self.cur_layer] = ((self.mask_sampler.sampling_params[f"mlp-{self.component_type}"][self.cur_layer][...,0] > 0) * 1).unsqueeze(0)

        self.compute_edges()

    def forward(self):
        prune_mask = self.prune_mask
        attn_unif = torch.rand((
            self.pruning_cfg.n_samples * self.pruning_cfg.batch_size, 
            *self.sampling_params[f"attn-{self.component_type}"][self.cur_layer].shape[:-1]
        )).to(self.pruning_cfg.device)
        prune_mask[f"attn-{self.component_type}"][self.cur_layer] = self.sample_prune_mask(
            attn_unif, 
            self.sampling_params[f"attn-{self.component_type}"][self.cur_layer]
        ).clone()

        mlp_unif = torch.rand((
            self.pruning_cfg.n_samples * self.pruning_cfg.batch_size, 
            *self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer].shape[:-1]
        )).to(self.pruning_cfg.device)
        prune_mask[f"mlp-{self.component_type}"][self.cur_layer] = self.sample_prune_mask(
            mlp_unif, 
            self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer]
        ).clone()
        
        return prune_mask, self.get_sampling_params()
