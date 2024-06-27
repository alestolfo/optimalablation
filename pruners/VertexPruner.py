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
from pruners.Pruner import Pruner
from utils.training_utils import LinePlot

class VertexPruner(Pruner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, parallel_inference=True)
    
    def process_null_val(self, node_type, layer_no):
        if node_type == "attn":
            null_val = self.null_vals['attn'][...,layer_no,:,:]
        elif node_type == "mlp":
            null_val = self.null_vals['mlp'][...,layer_no,:]
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
            attentions[bsz:] = (
                (prune_mask < 0.001) * (1-prune_mask) * null_val
                + (prune_mask >= 0.001) * (1-prune_mask) * null_val.detach()
            ) + prune_mask * attentions[bsz:].clone()
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
            mlp_out[bsz:] = (
                (prune_mask < 0.001) * (1-prune_mask) * null_val
                + (prune_mask >= 0.001) * (1-prune_mask) * null_val.detach()
            ) + prune_mask * mlp_out[bsz:].clone()

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
        
        return {
                **{f"patch_attn_{layer_no}": (partial(attention_points_filter, layer_no), 
                   partial(self.pruning_hook_attention_all_tokens, layer_no)
                ) for layer_no in range(n_layers)},
                **{f"patch_mlp_{layer_no}": (partial(mlp_out_filter, layer_no), 
                   partial(self.pruning_hook_mlp_all_tokens, layer_no)
                ) for layer_no in range(n_layers)},
                "patch_final": (final_embed_filter, self.final_hook_last_token)
        }