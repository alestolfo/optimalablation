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
from ..vertex_pruning.VertexPruner import VertexPruner
from ..mask_samplers.MaskSampler import SingleComponentMaskSampler, MultiComponentMaskSampler

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
pruning_cfg.n_samples = 10

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

high_losses = torch.zeros((1,)).to(device)

max_batches = 100

batch, last_token_pos = task_ds.next_batch(tokenizer)
for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):
    last_token_pos = last_token_pos.int()

    with torch.no_grad():
        loss = vertex_pruner(batch, last_token_pos)

        high_idx = (loss > .01).nonzero()
        high_losses = torch.cat([high_losses, loss[high_idx[:,0], high_idx[:,1]].flatten()], dim=0)

        head_losses, head_vars = update_means_variances(head_losses, head_vars, loss, no_batches)

    if no_batches % -10 == -1:
        sns.scatterplot(
            x=head_losses.cpu().flatten(), 
            y=head_vars.sqrt().cpu().flatten()
        )
        plt.xlabel("attention head mean loss")
        plt.ylabel("attention head std loss")
        plt.show()
# %%
torch.save({"head_loss": head_losses.unflatten(0, (n_layers, -1)), "head_var": head_vars.unflatten(0, (n_layers, -1))}, f"{folder}/mean_ablation_loss.pkl")
    

# sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
# modal_optimizer = torch.optim.AdamW([vertex_pruner.modal_attention, vertex_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

# get mean ablation loss
# back-prop: 
# %%
