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
from ..utils.MaskConfig import VertexInferenceConfig
from ..utils.training_utils import load_model_data, LinePlot, update_means_variances_mixed
from ..utils.task_datasets import IOIConfig, GTConfig
from ..mask_samplers.MaskSampler import MultiComponentMaskSampler, ConstantMaskSampler
from ..vertex_pruning.VertexPruner import VertexPruner

# %%

model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
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
# %%
pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.batch_size = 1
pruning_cfg.n_samples = 400

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = MultiComponentMaskSampler(pruning_cfg, prop_sample=0.001)
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
vertex_pruner.add_patching_hooks()
# vertex_pruner.cache_ablations = True

vertex_pruner.modal_attention.requires_grad = False
vertex_pruner.modal_mlp.requires_grad = False

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)

# %%

head_losses = torch.zeros((n_layers * n_heads,1)).to(device)
head_vars = torch.zeros((n_layers * n_heads,1)).to(device)
n_batches_by_head = torch.zeros_like(head_losses).to(device)
n_samples_by_head = torch.zeros_like(head_losses).to(device)

high_losses = torch.zeros((1,)).to(device)

max_batches = 10

batch, last_token_pos = task_ds.next_batch(tokenizer)
for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):
    last_token_pos = last_token_pos.int()

    sampling_optimizer.zero_grad()

    loss = vertex_pruner(batch, last_token_pos)
    loss.backward()

    atp_losses = torch.cat([ts.grad for ts in mask_sampler.mask_perturb['attn']], dim=0).unsqueeze(-1)

    batch_n_samples = []
    for ts in mask_sampler.sampled_mask['attn']:
        batch_n_samples.append((ts < 1-1e-3).sum(dim=0))
    batch_n_samples = torch.cat(batch_n_samples, dim=0).unsqueeze(-1)

    atp_losses = torch.where(
        batch_n_samples > 0,
        atp_losses / batch_n_samples,
        0
    )

    head_losses, head_vars, n_batches_by_head, n_samples_by_head = update_means_variances_mixed(head_losses, head_vars, atp_losses, n_batches_by_head, n_samples_by_head, batch_n_samples)

    if no_batches % -100 == -1:
        sns.scatterplot(
            x=head_losses.cpu().flatten(), 
            y=head_vars.sqrt().cpu().flatten()
        )
        plt.xlabel("attention head mean loss")
        plt.ylabel("attention head std loss")
        plt.show()
# %%
torch.save({"head_loss": head_losses.unflatten(0, (n_layers, -1)), "head_var": head_vars.unflatten(0, (n_layers, -1))}, f"{folder}/atp_gradsamp_loss.pkl")

# %%

mean_ablation_stats = torch.load(f"{folder}/mean_ablation_loss.pkl")
# %%
sns.scatterplot(y=mean_ablation_stats['head_loss'].cpu().flatten(), x=head_losses.cpu().flatten())
# %%

from circuit_utils import vertex_prune_mask
vertex_prune_mask['attn'][9][0,9] = 0

pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.batch_size = 1
pruning_cfg.n_samples = 20

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

mask_sampler = ConstantMaskSampler()

# %%
sampled_mask = {}
for k in vertex_prune_mask:
    sampled_mask[k] = []
    for ts in vertex_prune_mask[k]:
        sampled_mask[k].append(ts.repeat(pruning_cfg.n_samples,1).squeeze())

sampled_mask['attn'][9][:,9] = torch.arange(pruning_cfg.n_samples) / pruning_cfg.n_samples
sampled_mask['attn'][9] = torch.nn.Parameter(sampled_mask['attn'][9])

mask_sampler.sampled_mask = sampled_mask

my_optim = torch.optim.Adam([sampled_mask['attn'][9]], lr=0.01)
# %%
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
vertex_pruner.add_patching_hooks()

vertex_pruner.modal_attention.requires_grad = False
vertex_pruner.modal_mlp.requires_grad = False

# %%

head_losses = torch.zeros((n_layers * n_heads,1)).to(device)
head_vars = torch.zeros((n_layers * n_heads,1)).to(device)
n_batches_by_head = torch.zeros_like(head_losses).to(device)
n_samples_by_head = torch.zeros_like(head_losses).to(device)

high_losses = torch.zeros((1,)).to(device)

max_batches = 3

batch, last_token_pos = task_ds.next_batch(tokenizer)
for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):
    last_token_pos = last_token_pos.int()

    # sampling_optimizer.zero_grad()
    my_optim.zero_grad()

    loss = vertex_pruner(batch, last_token_pos)

    print(loss)
    loss.backward()
    
    # atp_losses = torch.cat([ts.grad for ts in mask_sampler.mask_perturb['attn']], dim=0).unsqueeze(-1)

    # batch_n_samples = []
    # for ts in mask_sampler.sampled_mask['attn']:
    #     batch_n_samples.append((ts < 1-1e-3).sum(dim=0))
    # batch_n_samples = torch.cat(batch_n_samples, dim=0).unsqueeze(-1)

    # atp_losses = torch.where(
    #     batch_n_samples > 0,
    #     atp_losses / batch_n_samples,
    #     0
    # )

    # head_losses, head_vars, n_batches_by_head, n_samples_by_head = update_means_variances_mixed(head_losses, head_vars, atp_losses, n_batches_by_head, n_samples_by_head, batch_n_samples)

    # if no_batches % -100 == -1:
    #     sns.scatterplot(
    #         x=head_losses.cpu().flatten(), 
    #         y=head_vars.sqrt().cpu().flatten()
    #     )
    #     plt.xlabel("attention head mean loss")
    #     plt.ylabel("attention head std loss")
    #     plt.show()

# %%
