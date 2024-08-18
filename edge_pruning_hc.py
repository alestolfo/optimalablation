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

# node_reg has same units as reg_lamb
# at peak, node_reg adds 50% more regularization to each edge
node_reg = min(0.5 * reg_lamb, 2e-4)

gpu_requeue = True

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
mask_sampler = EdgeMaskJointSampler(pruning_cfg, node_reg=node_reg)
edge_pruner = EdgePruner(model, pruning_cfg, mask_sampler, **task_ds.get_pruner_args())
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

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