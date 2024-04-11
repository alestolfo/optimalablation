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
from MaskSampler import SingleComponentMaskSampler, MultiComponentMaskSampler
from VertexPruner import VertexPruner
from MaskConfig import VertexInferenceConfig
from task_datasets import IOIConfig, GTConfig

# %%

model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.train()
# model.cfg.use_attn_result = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subfolder',
                        help='where to save stuff')
    args = parser.parse_args()
    subfolder = args.subfolder
except:
    subfolder = None

if subfolder is not None:
    folder=f"atp/{subfolder}"
else:
    folder=f"atp/ioi"

if not os.path.exists(folder):
    os.makedirs(folder)

pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.batch_size = 5
pruning_cfg.n_samples = n_layers * n_heads

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = SingleComponentMaskSampler(pruning_cfg)
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
vertex_pruner.add_patching_hooks()

# %%

head_losses = torch.zeros((pruning_cfg.n_samples,1)).to(device)
head_vars = torch.zeros((pruning_cfg.n_samples,1)).to(device)

max_batches = 1000

for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):
    batch, last_token_pos = task_ds.next_batch(tokenizer)
    last_token_pos = last_token_pos.int()

    with torch.no_grad():
        loss = vertex_pruner(batch, last_token_pos)

        # how to compute variance iteratively?
        head_vars = head_vars + (loss - head_losses).square().sum(dim=-1) - (loss.mean(dim=-1) - head_losses).square()
        head_losses = (no_batches * head_losses + loss.mean(dim=-1)) / (no_batches + 1)

    if no_batches % -100 == -1:
        sns.histplot(head_losses.cpu().flatten())
        sns.histplot((head_vars / no_batches).cpu().flatten())
    
    no_batches += 1
    break

# sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
# modal_optimizer = torch.optim.AdamW([vertex_pruner.modal_attention, vertex_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

# get mean ablation loss
# back-prop: 