# %%
import torch
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, LinePlot
import json
from pathlib import Path
from greater_than.utils import get_valid_years
from greater_than.data import YearDataset

# %%

def cache_hook_all_tokens(bsz, storage, activations, hook):
    if bsz is None:
        storage.append(activations)
        return activations
    else:
        storage.append(activations[bsz:])
        return activations[:bsz]

# NOTE: FOR BACKWARD-FACING ABLATION SET MLP_CONSTANTS TO NONE

# attention_constants: list of all constants for attention for layers thus far
# mlp_constants: list of all constants for embed+mlp layers thus far
# attention_cache: contains all attentions stored thus far, list of attention outputs by later
# mlp_cache: list of mlp outputs by layer
def pruning_edge_attention_hook_all_tokens(bsz, W_O, prune_mask, attn_constants, mlp_constants, attn_cache, mlp_cache, total_post_bias, orig_in, hook):
        
    def prepend_orig(out):
        if bsz is None:
            return out
        return torch.cat([orig_in[:bsz], out], dim=0)
    # i is the current layer (0-indexed, equal to the number of layers before this one)
    # orig_in: batch x seq_pos x d_model
    # prune_mask[0]: (bsz * n_samples) x n_heads (dest) x i x n_heads (source)
    # attention_constants: i x n_heads (source) x d_model
    # attention_cache: i * [(bsz * n_samples) x seq_pos x n_heads (source) x d_model]

    # mlp_constants: (i+1) x d_model
    # mlp_cache: (i+1) * [(bsz * n_samples) x seq_pos x d_model]

    # mlp_mask: (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x 1 (d_model)

    # print((orig_in - mlp_cache[0].unsqueeze(-2)).square().sum())
    # return orig_in
    mlp_mask = prune_mask[1].unsqueeze(1).unsqueeze(-1)

    # print((torch.stack(mlp_cache, dim=-2).unsqueeze(dim=2).sum(dim=-2) - orig_in).square().sum())

    # print(mlp_mask.shape)
    # print(mlp_cache[0].shape)
    # print(mlp_constants.shape)
    
    out = (mlp_mask * torch.stack(mlp_cache, dim=-2).unsqueeze(dim=2)).sum(dim=-2)
    # print((out-orig_in).square().sum())

    if mlp_constants is not None:
        out = out + ((1-mlp_mask) * mlp_constants).sum(dim=-2)

    if prune_mask[0] is None:
        return prepend_orig(out)
    
    # (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x n_heads (source) x 1 (d_model/d_head)
    attn_mask = prune_mask[0].unsqueeze(1).unsqueeze(-1)
    attn_term = attn_mask * torch.stack(attn_cache, dim=-3).unsqueeze(dim=2)

    # W_O: source_head x d_head x d_model
    if W_O is None:
        attn_term = attn_term.sum(dim=[-3,-2])
    else:
        attn_term = einsum(
                    "batch pos dest_head prev_layer source_head d_head, \
                        prev_layer source_head d_head d_model -> \
                        batch pos dest_head d_model",
                    attn_term,
                    W_O
            )
    out = out + attn_term + total_post_bias

    if mlp_constants is None:
        return prepend_orig(out + attn_constants)
    
    # torch.cuda.empty_cache()
    # print((out - orig_in).abs().mean())
    return prepend_orig(out + ((1-attn_mask) * attn_constants).sum(dim=[-3,-2]))

# NOTE: for backward-facing ablation set attn_constants to none

# same as attentions except not parallelized
# attention_constants: list of all constants for attention for layers thus far
# mlp_constants: list of all constants for embed+mlp layers thus far
# attention_cache: contains all attentions stored thus far, list of attention outputs by later
# mlp_cache: list of mlp outputs by layer
def pruning_edge_mlp_hook_all_tokens(bsz, W_O, prune_mask, attn_constants, mlp_constants, attn_cache, mlp_cache, total_post_bias, orig_in, hook):
    
    def prepend_orig(out):
        if bsz is None:
            return out
        return torch.cat([orig_in[:bsz], out], dim=0)
    # i is the current layer (0-indexed, equal to the number of layers before this one)
    # orig_in: batch x seq_pos x d_model
    # prune_mask[0]: (bsz * n_samples) x i x n_heads
    # attention_constants: i x n_heads x d_model
    # attention_cache: i * [(bsz * n_samples) x seq_pos x n_heads x d_model]

    # mlp_constants: (i+1) x d_model
    # mlp_cache: (i+1) * [(bsz * n_samples) x seq_pos x d_model]

    # (bsz * n_samples) x 1 (seq_pos) x i x 1 (d_model)
    mlp_mask = prune_mask[1].unsqueeze(1).unsqueeze(-1)

    out = (mlp_mask * torch.stack(mlp_cache, dim=2)).sum(dim=2)

    if attn_constants is not None:
        out = out + ((1-mlp_mask) * mlp_constants).sum(dim=2)

    # print((out - mlp_cache[0]).square().sum())

    if prune_mask[0] is None:
        return prepend_orig(out)
    
    # (bsz * n_samples) x 1 (seq_pos) x i x n_heads x 1 (d_model)
    attn_mask = prune_mask[0].unsqueeze(1).unsqueeze(-1)
    attn_term = attn_mask * torch.stack(attn_cache, dim=-3)

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
    
    # print(attn_term.shape)
    # print(attn_cache[0].shape)
    # print((attn_term - attn_cache[0].sum(dim=2)).square().sum())
    
    # print(orig_in.shape)
    # print(out.shape)
    # print((out - orig_in).abs().mean())
    # torch.cuda.empty_cache()
    
    return prepend_orig(out + ((1-attn_mask) * attn_constants).sum(dim=[2,3]))


def pruning_edge_final_hook_all_tokens(last_token_mask, bsz, parallel_inference, W_O, prune_mask, attn_constants, mlp_constants, attn_cache, mlp_cache, total_post_bias, orig_in, hook):
    prev_time = time.time()
    out = pruning_edge_mlp_hook_all_tokens(bsz if parallel_inference else None, W_O, prune_mask, attn_constants, mlp_constants, attn_cache, mlp_cache, total_post_bias, orig_in, hook).unflatten(0, (-1, bsz))


    out = (out * last_token_mask.unsqueeze(-1)).sum(dim=2)
    print("End", time.time()-prev_time)
    return out


# %%

# beta and alpha should be same shape as x, or broadcastable
# def f_concrete(x, beta, alpha):
#     return ((x.log() - (1-x).log()) * beta - alpha.log()).sigmoid()

def sample_prune_mask(unif, sampling_params, hard_concrete_endpoints):
    # back prop against log alpha
    concrete = (((.001+unif).log() - (1-unif).log() + sampling_params[...,0])/(sampling_params[...,1].relu()+.001)).sigmoid()

    hard_concrete = ((concrete + hard_concrete_endpoints[0]) * (hard_concrete_endpoints[1] - hard_concrete_endpoints[0])).clamp(0,1)

    # n_layers x (total_samples = batch_size * n_samples) x n_heads
    return hard_concrete

def get_cache_hooks(cache_compressed_attn, n_layers, attention_cache, mlp_cache, bsz):
    embed_filter = lambda name: name == f"blocks.{0}.hook_resid_pre"
    attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"
    attention_compressed_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_z"
    mlp_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"

    return [
        # cache embedding
        (embed_filter, 
        partial(cache_hook_all_tokens, bsz, mlp_cache)),

        # cache attention (at z if compressed)
        *[
            (partial(attention_compressed_filter if cache_compressed_attn else attention_points_filter, layer_no), 
            partial(cache_hook_all_tokens, bsz, attention_cache)) 
        for layer_no in range(n_layers)],

        # cache MLP
        *[
            (partial(mlp_points_filter, layer_no), 
            partial(cache_hook_all_tokens, bsz, mlp_cache)) 
        for layer_no in range(n_layers)],
    ]

def get_pruning_hooks(W_O, post_bias, n_layers, attention_cache, mlp_cache, modal_values, prune_mask, last_token_mask, bsz, parallel_inference=False, ablation_backward=False):
    circs = ["q", "k", "v"]
    attention_in_filter = lambda layer_no, circ, name: name == f"blocks.{layer_no}.hook_{circ}_input"
    mlp_in_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_in"
    final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

    return [
        # patch attention (recompute O-matrix if compressed)
        *[(partial(attention_in_filter, layer_no, circ), 
            partial(pruning_edge_attention_hook_all_tokens,
                    bsz if parallel_inference else None, 
                    None if W_O is None else W_O[:layer_no], 
                    # prune_mask[layer_no][j], 
                    [prune_mask["attn-attn"][layer_no][:,j] if layer_no > 0 else None, 
                        prune_mask["mlp-attn"][layer_no][:,j]], 
                    modal_values[0][layer_no] if ablation_backward else modal_values[0][:layer_no], 
                    None if ablation_backward else modal_values[1][:layer_no+1], 
                    attention_cache, 
                    mlp_cache, 
                    post_bias[:layer_no].sum(dim=0))) 
        for layer_no in range(n_layers) for j, circ in enumerate(circs)],

        # patch MLP (recompute O-matrix if compressed)
        *[(partial(mlp_in_filter, layer_no), 
            partial(pruning_edge_mlp_hook_all_tokens, 
                    bsz if parallel_inference else None,
                    None if W_O is None else W_O[:layer_no+1], 
                    # prune_mask[layer_no][-1], 
                    [prune_mask["attn-mlp"][layer_no], prune_mask["mlp-mlp"][layer_no]], 
                    None if ablation_backward else modal_values[0][:layer_no+1], 
                    modal_values[1][layer_no] if ablation_backward else modal_values[1][:layer_no+1], 
                    attention_cache,
                    mlp_cache, 
                    post_bias[:layer_no+1].sum(dim=0))) 
        for layer_no in range(n_layers)],

        # patch MLP (recompute O-matrix if compressed)
        (final_embed_filter, 
            partial(pruning_edge_final_hook_all_tokens, 
                    last_token_mask,
                    bsz,
                    parallel_inference,
                    None if W_O is None else W_O, 
                    # prune_mask[-1], 
                    [prune_mask["attn-mlp"][-1], prune_mask["mlp-mlp"][-1]], 
                    None if ablation_backward else modal_values[0], 
                    modal_values[1][-1] if ablation_backward else modal_values[1], 
                    attention_cache,
                    mlp_cache, 
                    post_bias.sum(dim=0)))
    ]

# %%

