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
from utils.training_utils import load_model_data, load_args, update_means_variances, update_means_variances_mixed, update_means_variances_exponential, plot_no_outliers
from utils.MaskConfig import VertexInferenceConfig
from utils.task_datasets import get_task_ds
from pruners.VertexPruner import VertexPruner
from mask_samplers.AblationMaskSampler import SingleComponentMaskSampler

# %%

model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
# model.cfg.use_attn_result = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
# desc: ablation type. Supported ablation types: zero, mean, oca, resample, refmean, cf
args = load_args("ablation_loss", defaults={"desc": "resample", "dataset": "gt"})
folder, ablation_type, dataset = args["folder"], args["desc"], args["dataset"]

pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.batch_size = 3

task_ds = get_task_ds(dataset, pruning_cfg.batch_size, device, fix_prompt=(ablation_type == "resample"))

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = SingleComponentMaskSampler(pruning_cfg)
pruning_cfg.n_samples = mask_sampler.n_components

oca_train = False

if ablation_type == "zero":
    zero_modes = []
    init_modes = task_ds.init_modes()
    for ts in init_modes:
        zero_modes.append(torch.zeros_like(ts).to(device))
    init_modes = zero_modes
elif ablation_type == "refmean":
    init_modes = task_ds.init_modes(True)

# training mode if there are no recorded modes
elif ablation_type == "oca":
    oca_train = not os.path.exists(f"{folder}/oca_modes.pth")
    if not oca_train:
        init_modes = torch.load(f"{folder}/oca_modes.pth")
        init_modes = init_modes['modal_attention'], init_modes['modal_mlp']
else:
    init_modes = task_ds.init_modes()


cf_mode = ablation_type in {"resample", "cf"}
vertex_pruner = VertexPruner(model, pruning_cfg, init_modes, mask_sampler, counterfactual_mode=cf_mode)
vertex_pruner.add_patching_hooks()

# %%

if oca_train:
    max_batches = 10000
    modal_optimizer = torch.optim.AdamW([vertex_pruner.modal_attention, vertex_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)
else:
    max_batches = 1000
    head_losses = torch.zeros((pruning_cfg.n_samples,1)).to(device)
    head_vars = torch.zeros((pruning_cfg.n_samples,1)).to(device)

for no_batches in tqdm(range(max_batches)):
    if ablation_type == "cf":
        batch, last_token_pos, cf = task_ds.next_batch(tokenizer, counterfactual=True)
    else:
        batch, last_token_pos = task_ds.next_batch(tokenizer)

    if ablation_type == "resample":
        cf = batch[torch.randperm(batch.shape[0]).to(device)]
    last_token_pos = last_token_pos.int()

    if oca_train:
        modal_optimizer.zero_grad()

        loss = vertex_pruner(batch, last_token_pos)
        loss.backward()
        modal_optimizer.step()

        if no_batches % -1000 == -1:
            torch.save({"modal_attention": vertex_pruner.modal_attention, "modal_mlp": vertex_pruner.modal_mlp}, f"{folder}/oca_modes.pth")
            vertex_pruner.log.plot(["kl_loss"], mv=100, save=f"{folder}/oca_train.png")
    else:
        with torch.no_grad():
            # loss: [n_components, batch_size]
            if cf_mode:
                loss, _ = vertex_pruner(batch, last_token_pos, separate_loss=True, counterfactual=cf)
            else:
                loss, _ = vertex_pruner(batch, last_token_pos, separate_loss=True)

            head_losses, head_vars = update_means_variances(head_losses, head_vars, loss, no_batches)

if not oca_train:
    torch.save({"head_losses": head_losses, "head_vars": head_vars}, f"{folder}/{ablation_type}_results.pth")

    plot_no_outliers(
        sns.scatterplot, .03,
        head_losses, head_vars.sqrt(),
        args={"x": "Component mean loss", "y": "Component std loss", "s": 5,
              "f": f"{folder}/{ablation_type}.png"}
    )

# %%

# mean_diff = []
# var_diff = []

# head_losses = torch.zeros((pruning_cfg.n_samples,1)).to(device)
# head_vars = torch.zeros((pruning_cfg.n_samples,1)).to(device)
# n_samples = torch.zeros((pruning_cfg.n_samples,1)).to(device)
# n_bbyhead = torch.zeros((pruning_cfg.n_samples,1)).to(device)

# y = all_losses.shape[1]
# for x in range(y):
#     # print(head_losses.shape)
#     # print(((x * head_losses + all_losses[:,x].mean(dim=-1, keepdim=True)) / (x + 1)).mean())
#     head_losses, head_vars, n_bbyhead, n_samples = update_means_variances_exponential(head_losses, head_vars, all_losses[:,[x]], n_bbyhead, n_samples, torch.ones_like(all_losses[:,[x]]).to(device), x)
#     # head_losses, head_vars, n_bbyhead, n_samples = update_means_variances_mixed(head_losses, head_vars, all_losses[:,[x]], n_bbyhead, n_samples, torch.ones_like(all_losses[:,[x]]).to(device))
#     # print(head_losses.shape)
#     # print(head_losses)
#     print("Crazy loss", all_losses[144,x])
#     print("True variance", all_losses[:,max(0,x-20):x+1].var(dim=-1)[144])
#     mean_error = (all_losses[:,:x+1].mean(dim=1) - head_losses.squeeze(-1)).abs()
#     mean_error[144] = 0
#     print(mean_error.max())
#     mean_diff.append(mean_error.mean().item())
#     # if x == 11:
#     #     head_losses = all_losses[:,:x+1].mean(dim=-1).unsqueeze(-1)
#     #     head_vars = all_losses[:,:x+1].var(dim=-1).unsqueeze(-1)
#     var_error = (all_losses[:,:x+1].var(dim=-1) - head_vars.squeeze(-1)).abs()
#     var_error[144] = 0
#     print(var_error.max())
#     var_diff.append(var_error.mean().item())
#     # print(mean_diff)
#     # print(var_diff)

# sns.lineplot(mean_diff)
# sns.lineplot(var_diff)
# # %%
# torch.save({"head_loss": head_losses.unflatten(0, (n_layers, -1)), "head_var": head_vars.unflatten(0, (n_layers, -1))}, f"{folder}/mean_ablation_loss.pkl")
    

# sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
# modal_optimizer = torch.optim.AdamW([vertex_pruner.modal_attention, vertex_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

# get mean ablation loss
# back-prop: 
# %%
