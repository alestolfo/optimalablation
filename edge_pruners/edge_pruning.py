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
from mask_samplers.MaskSampler import EdgeMaskJointSampler
from utils.MaskConfig import EdgeInferenceConfig
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
    args = parser.parse_args()
    reg_lamb = float(args.lamb)
    subfolder = args.subfolder
except:
    reg_lamb = None
    subfolder = None

if reg_lamb is None:
    reg_lamb = 2e-4

node_reg=5e-3
gpu_requeue = True
# reset_optim = 1000

print(reg_lamb)

if subfolder is not None:
    folder=f"pruning_edges_auto/ioi_edges/{subfolder}"
else:
    folder=f"pruning_edges_auto/ioi_edges/{reg_lamb}"

pretrained_folder = None
# f"pruning_edges_auto/ioi/300.0"
if not os.path.exists(folder):
    os.makedirs(folder)

pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, init_param=0)
pruning_cfg.lamb = reg_lamb

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%

# prune_retrain = True
# prune_length = 200
# retrain_length = 100

# %%
mask_sampler = EdgeMaskJointSampler(pruning_cfg, node_reg=node_reg)
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)

take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)
# %%
# if prune_retrain and edge_pruner.log.t == 0:
#     edge_pruner.log.mode = "prune"
#     edge_pruner.log.cur_counter = 0

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

    grad_norms = []
    for k in mask_sampler.sampling_params:
        grad_norm = torch.nn.utils.clip_grad_norm_(mask_sampler.sampling_params[k], max_norm=float('inf'))
        grad_norms.append(grad_norm.item())
        torch.nn.utils.clip_grad_norm_(mask_sampler.sampling_params[k], max_norm=5)
    print(grad_norms)

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
        lp_count.add_entry({
            "step_size": step_sz.item(), 
            "mode_step_size": mode_step_sz.item(),
            "max_grad_norm": np.max(grad_norms)
        })

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
