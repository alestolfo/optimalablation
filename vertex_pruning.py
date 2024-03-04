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
import os
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, LinePlot
from MaskSampler import MaskSampler
from VertexPruner import VertexPruner
from MaskConfig import VertexInferenceConfig
from task_datasets import IOIConfig, GTConfig

# %%

model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.train()
model.cfg.use_attn_result = True
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
    reg_lamb = 3e-3

gpu_requeue = True

print(reg_lamb)

if subfolder is not None:
    folder=f"pruning_vertices_auto/ioi_with_mlp/{subfolder}"
else:
    folder=f"pruning_vertices_auto/ioi_with_mlp/{reg_lamb}"

pretrained_folder = None
# f"pruning_edges_auto/ioi/300.0"
if not os.path.exists(folder):
    os.makedirs(folder)

pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.lamb = reg_lamb
pruning_cfg.lr_modes = 5e-3

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = MaskSampler(pruning_cfg)
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
vertex_pruner.add_patching_hooks()

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
modal_optimizer = torch.optim.AdamW([vertex_pruner.modal_attention, vertex_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

lp_count = pruning_cfg.load_snapshot(vertex_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)

take_snapshot = partial(pruning_cfg.take_snapshot, vertex_pruner, lp_count, sampling_optimizer, modal_optimizer)

# %%

max_batches = 6000
for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):

    plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
    checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    last_token_pos = last_token_pos.int()

    modal_optimizer.zero_grad()
    sampling_optimizer.zero_grad()

    # sample prune mask
    graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
    loss, all_sampling_params = vertex_pruner(batch, last_token_pos, graph_suffix)
    loss.backward()

    prev_alphas = all_sampling_params[:,0].detach().clone()
    prev_modes = vertex_pruner.get_modes().detach().clone()

    sampling_optimizer.step()
    modal_optimizer.step()

    mask_sampler.fix_nans()

    with torch.no_grad():
        step_sz = (mask_sampler.get_sampling_params()[:,0] - prev_alphas).abs()
        step_sz = (step_sz - 1e-3).relu().sum() / (step_sz > 1e-3).sum()
        mode_step_sz = (vertex_pruner.get_modes().clone() - prev_modes).norm(dim=-1).mean()
        lp_count.add_entry({"step_size": step_sz.item(), "mode_step_size": mode_step_sz.item()})

    if plotting:
        take_snapshot("")
        if checkpointing:
            take_snapshot(f"-{no_batches}")
        if vertex_pruner.early_term(.005) >= 10:
            take_snapshot("-final")
            break
# %%

vertex_pruner
# %%
