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
from ..utils.training_utils import load_model_data, LinePlot, update_means_variances
from ..utils.MaskConfig import VertexInferenceConfig
from ..utils.task_datasets import IOIConfig, GTConfig
from ..mask_samplers.MaskSampler import AttributionPatchingMaskSampler, MultiComponentMaskSampler
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

pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.batch_size = 20
pruning_cfg.n_samples = 1

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = AttributionPatchingMaskSampler(pruning_cfg)
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
vertex_pruner.add_patching_hooks()
# vertex_pruner.cache_ablations = True

vertex_pruner.modal_attention.requires_grad = False
vertex_pruner.modal_mlp.requires_grad = False

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)

# %%

head_losses = torch.zeros((pruning_cfg.n_samples,1)).to(device)
head_vars = torch.zeros((pruning_cfg.n_samples,1)).to(device)

high_losses = torch.zeros((1,)).to(device)

max_batches = 100

batch, last_token_pos = task_ds.next_batch(tokenizer)

for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):
    last_token_pos = last_token_pos.int()

    sampling_optimizer.zero_grad()

    loss = vertex_pruner(batch, last_token_pos)
    mean_loss = loss.mean()
    mean_loss.backward()

    sampling_optimizer.step()

    atp_losses = torch.cat([ts.grad for ts in mask_sampler.sampling_params['attn']], dim=0).unsqueeze(-1)

    # NOTE: since i'm differentiating wrt a single parameter for the entire batch
    # the variance over individual samples is batch_size * head_vars
    # since head_vars reflects the variance of the estimated batch means.
    head_losses, head_vars = update_means_variances(head_losses, head_vars, atp_losses, no_batches)

    if no_batches % -100 == -1:
        sns.scatterplot(
            x=head_losses.cpu().flatten(), 
            y=(pruning_cfg.batch_size * head_vars).sqrt().cpu().flatten()
        )
        plt.xlabel("attention head mean loss")
        plt.ylabel("attention head std loss")
        plt.show()
# %%
torch.save({"head_loss": head_losses.unflatten(0, (n_layers, -1)), "head_var": head_vars.unflatten(0, (n_layers, -1))}, f"{folder}/atp_loss.pkl")
# %%
