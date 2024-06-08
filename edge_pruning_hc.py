# %%
import torch
import datasets
import os
from sys import argv
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
import argparse
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pruners.EdgePruner import EdgePruner
from utils.circuit_utils import retrieve_mask, edge_prune_mask, discretize_mask
from mask_samplers.EdgeMaskSampler import EdgeMaskJointSampler
from utils.MaskConfig import EdgeInferenceConfig
from utils.task_datasets import get_task_ds
from utils.training_utils import load_model_data, LinePlot, load_args

# %%
# load model
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.train()
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
args = load_args("pruning", 1.5e-3, {
    "desc": "cf",
    "name": "hc",
    "window": False
})

folder, reg_lamb, dataset, prior_lamb, prior_scale, run_name, ablation_type = args["folder"], args["lamb"], args["dataset"], args["priorlamb"], args["priorscale"], args["name"], args["desc"]
print("Folder", folder)
print("Lamb", reg_lamb)
print("Dataset", dataset)
print("Ablation type", ablation_type)

cf_mode = ablation_type in {"resample", "cf"}

# node_reg has same units as reg_lamb
# at peak, node_reg adds 50% more regularization to each edge
node_reg = min(0.5 * reg_lamb, 2e-4)

gpu_requeue = True
# reset_optim = 1000

pretrained_folder = None

init_param = 0
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, init_param=init_param, use_temp=True)
pruning_cfg.lamb = reg_lamb

if reg_lamb <= 1e-4:
    pruning_cfg.lr = 1.5e-1
elif reg_lamb <= 5e-4:
    pruning_cfg.lr = 1e-1
else:
    pruning_cfg.lr = 5e-2

if ablation_type == "cf":
    pruning_cfg.lr /= 3
elif ablation_type == "resample":
    pruning_cfg.lr /= 1.5

task_ds = get_task_ds(dataset, pruning_cfg.batch_size, device, ablation_type)

for param in model.parameters():
    param.requires_grad = False

# %%
if run_name == "vertex_prior":
    prior_folder = f"results/pruning_vertices_auto/{dataset}/{prior_lamb}"
    vertex_prune_mask, state_dict = retrieve_mask(prior_folder, state_dict=True)
    all_alphas = torch.cat([ts.flatten() for k in vertex_prune_mask for ts in vertex_prune_mask[k]], dim=0)
    sns.histplot(all_alphas.cpu())

    vertex_prune_mask['mlp'].insert(0, torch.tensor([0]).to(device))
    vertex_prune_mask['mlp'].append(torch.tensor([0]).to(device))

    for i, ts in enumerate(edge_prune_mask['attn-attn']):
        if i == 0:
            continue
        src_contrib = torch.stack(vertex_prune_mask['attn'][:i],dim=1).squeeze(0)
        dest_contrib = vertex_prune_mask['attn'][i].unsqueeze(-1).unsqueeze(-1)
        ts *= 0 
        ts += init_param + prior_scale * (src_contrib + dest_contrib)

    for i, ts in enumerate(edge_prune_mask['mlp-attn']):
        src_contrib = torch.stack(vertex_prune_mask['mlp'][:i+1],dim=1).squeeze(0)
        dest_contrib = vertex_prune_mask['attn'][i].unsqueeze(-1)
        ts *= 0 
        ts += init_param + prior_scale * (src_contrib + dest_contrib)

    for i, ts in enumerate(edge_prune_mask['attn-mlp']):
        src_contrib = torch.stack(vertex_prune_mask['attn'][:i+1],dim=1).squeeze(0)
        dest_contrib = vertex_prune_mask['mlp'][i+1].unsqueeze(-1).unsqueeze(-1)
        ts *= 0 
        ts += init_param + prior_scale * (src_contrib + dest_contrib)

    for i, ts in enumerate(edge_prune_mask['mlp-mlp']):
        src_contrib = torch.stack(vertex_prune_mask['mlp'][:i+1],dim=1).squeeze(0)
        dest_contrib = vertex_prune_mask['mlp'][i+1].unsqueeze(-1)
        ts *= 0 
        ts += init_param + prior_scale * (src_contrib + dest_contrib)

    pruning_cfg.constant_prune_mask = edge_prune_mask
    pruning_cfg.initialize_params(1, use_temp=True)

# %%
mask_sampler = EdgeMaskJointSampler(pruning_cfg, node_reg=node_reg)
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, counterfactual_mode=cf_mode)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

if run_name == "vertex_prior":
    edge_pruner.load_state_dict(state_dict, strict=False)

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)

if ablation_type == "oa":
    modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)
else:
    modal_optimizer = None
    if ablation_type == "mean":
        edge_pruner.modal_attention.requires_grad = False
        edge_pruner.modal_mlp.requires_grad = False
# %%

lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)

take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)

# if prune_retrain and edge_pruner.log.t == 0:
#     edge_pruner.log.mode = "prune"
#     edge_pruner.log.cur_counter = 0

# %%
if ablation_type == "oa":
    max_batches = 6000
else:
    max_batches = 3000
    
for no_batches in tqdm(range(edge_pruner.log.t, max_batches)):

    plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
    checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

    batch, last_token_pos, cf = task_ds.retrieve_batch_cf(tokenizer)

    sampling_optimizer.zero_grad()
    if ablation_type == "oa":
        modal_optimizer.zero_grad()

    # sample prune mask
    graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
    loss = edge_pruner(batch, last_token_pos, cf, graph_suffix=graph_suffix)
    loss.backward()

    grad_norms = []
    for k in mask_sampler.sampling_params:
        grad_norm = torch.nn.utils.clip_grad_norm_(mask_sampler.sampling_params[k], max_norm=float('inf'))
        grad_norms.append(grad_norm.item())
        torch.nn.utils.clip_grad_norm_(mask_sampler.sampling_params[k], max_norm=5)
    print(grad_norms)

    prev_alphas = mask_sampler.get_sampling_params()[:,0].detach().clone()

    # if prune_retrain:
    #     edge_pruner.log.cur_counter += 1
    #     if edge_pruner.log.mode == "prune":
    #         sampling_optimizer.step()

    #         if edge_pruner.log.cur_counter >= prune_length:
    #             edge_pruner.log.cur_counter = 0
    #             edge_pruner.log.mode = "retrain"
    #             modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

    #     elif edge_pruner.log.cur_counter >= retrain_length:
    #         edge_pruner.log.cur_counter = 0
    #         edge_pruner.log.mode = "prune"
    # else:
    sampling_optimizer.step()

    if ablation_type == "oa":
        prev_modes = edge_pruner.get_modes().detach().clone()
        modal_optimizer.step()

    mask_sampler.fix_nans()

    with torch.no_grad():
        step_sz = (mask_sampler.get_sampling_params()[:,0] - prev_alphas).abs()
        step_sz = (step_sz - 1e-3).relu().sum() / (step_sz > 1e-3).sum()
        lp_entry = {
            "step_size": step_sz.item(), 
            "max_grad_norm": np.max(grad_norms)
        }

        if ablation_type == "oa":
            mode_step_sz = (edge_pruner.get_modes().clone() - prev_modes).norm(dim=-1).mean()
            lp_entry["mode_step_size"] = mode_step_sz.item()
            
        lp_count.add_entry(lp_entry)

    if plotting:
        take_snapshot("")
        if checkpointing:
            take_snapshot(f"-{no_batches}")
        # if edge_pruner.early_term() >= 10:
        #     take_snapshot("-final")
        #     break
    
    # if reset_optim is not None and no_batches % (-1 * reset_optim) == -1:
    #     modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)
# %%
