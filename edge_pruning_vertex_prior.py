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
from EdgePruner import EdgePruner
from VertexPruner import VertexPruner
from circuit_utils import retrieve_mask, edge_prune_mask, discretize_mask
from mask_samplers.MaskSampler import MaskSampler, EdgeMaskJointSampler
from MaskConfig import EdgeInferenceConfig
from task_datasets import IOIConfig, GTConfig
from training_utils import load_model_data, LinePlot

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

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lamb',
                        help='regularization constant')
    parser.add_argument('-s', '--subfolder',
                        help='where to save stuff')
    parser.add_argument('-t', '--strength',
                        help='prior strength')
    parser.add_argument('-p', '--prior',
                        help='which vertex lambda')
    args = parser.parse_args()
    reg_lamb = float(args.lamb)
    prior_lamb = float(args.prior)
    prior_scale = float(args.strength)
    prior_lamb = float(args.prior)
    subfolder = args.subfolder
except:
    raise Exception()
    reg_lamb = None
    subfolder = None
    prior_scale = 0.5
    prior_lamb = 7e-3

if reg_lamb is None:
    reg_lamb = 4e-4

node_reg=5e-3
gpu_requeue = True
# reset_optim = 1000

print(reg_lamb)

if subfolder is not None:
    folder=f"pruning_edges_auto/ioi_vertex_prior/{subfolder}"
else:
    folder=f"pruning_edges_auto/ioi_vertex_prior/{reg_lamb}-{prior_lamb}-{prior_scale}"

prior_folder = f"pruning_vertices_auto/ioi_with_mlp/{prior_lamb}" 

pretrained_folder = None
# f"pruning_edges_auto/ioi/300.0"
if not os.path.exists(folder):
    os.makedirs(folder)

init_param = 0
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, init_param=init_param)
pruning_cfg.lamb = reg_lamb

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
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
pruning_cfg.initialize_params(1,None)
# %%
# prune_retrain = True
# prune_length = 200
# retrain_length = 100

# %%
mask_sampler = EdgeMaskJointSampler(pruning_cfg, node_reg=node_reg)
edge_pruner = EdgePruner(model, pruning_cfg, [state_dict["modal_attention"], state_dict["modal_mlp"]], mask_sampler, parallel_inference=True)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)

take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)

# if prune_retrain and edge_pruner.log.t == 0:
#     edge_pruner.log.mode = "prune"
#     edge_pruner.log.cur_counter = 0

# %%
max_batches = 10000
for no_batches in tqdm(range(edge_pruner.log.t, max_batches)):

    plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
    checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    last_token_pos = last_token_pos.int()

    modal_optimizer.zero_grad()
    sampling_optimizer.zero_grad()

    # sample prune mask
    graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
    loss = edge_pruner(batch, last_token_pos, graph_suffix)
    loss.backward()

    prev_alphas = mask_sampler.get_sampling_params()[:,0].detach().clone()
    prev_modes = edge_pruner.get_modes().detach().clone()

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
    modal_optimizer.step()

    mask_sampler.fix_nans()

    with torch.no_grad():
        step_sz = (mask_sampler.get_sampling_params()[:,0] - prev_alphas).abs()
        step_sz = (step_sz - 1e-3).relu().sum() / (step_sz > 1e-3).sum()
        mode_step_sz = (edge_pruner.get_modes().clone() - prev_modes).norm(dim=-1).mean()
        lp_count.add_entry({"step_size": step_sz.item(), "mode_step_size": mode_step_sz.item()})

    if plotting:
        take_snapshot("")
        if checkpointing:
            take_snapshot(f"-{no_batches}")
        if edge_pruner.early_term() >= 10:
            take_snapshot("-final")
            break
    
    # if reset_optim is not None and no_batches % (-1 * reset_optim) == -1:
    #     modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)
# %%
