# %%
import torch
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
from utils.training_utils import load_model_data, load_args, update_means_variances, plot_no_outliers
from utils.MaskConfig import VertexInferenceConfig
from utils.task_datasets import get_task_ds
from pruners.VertexPruner import VertexPruner
from mask_samplers.AblationMaskSampler import SingleComponentMaskSampler

# %%

model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
# desc: ablation type. Supported ablation types: zero, mean, oa, resample, cf_mean, cf
args = load_args("ablation_loss", defaults={"desc": "oa_specific", "dataset": "ioi"})
folder, ablation_type, dataset = args["folder"], args["desc"], args["dataset"]

pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.batch_size = 10

oa_train = False
if ablation_type.startswith("oa"):
    if os.path.exists(f"{folder}/{ablation_type}_modes.pth"):
        init_modes = torch.load(f"{folder}/{ablation_type}_modes.pth")
    else:
        pruning_cfg.batch_size = 3
        oa_train = True

task_ds = get_task_ds(dataset, pruning_cfg.batch_size, device, ablation_type)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = SingleComponentMaskSampler(pruning_cfg)
pruning_cfg.n_samples = mask_sampler.n_components

pruner_args = task_ds.get_pruner_args({"zero", "mean", "resample", "cf_mean", "cf", "oa", "oa_specific"})

if ablation_type.startswith("oa") and not oa_train:
    pruner_args['init_modes'] = init_modes['modal_attention'], init_modes['modal_mlp']

vertex_pruner = VertexPruner(model, pruning_cfg, mask_sampler, **pruner_args)
vertex_pruner.add_patching_hooks()

# %%

if oa_train:
    max_batches = 10000
    modal_optimizer = torch.optim.AdamW([vertex_pruner.modal_attention, vertex_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)
else:
    max_batches = 10000 // pruning_cfg.batch_size
    head_losses = torch.zeros((pruning_cfg.n_samples,1)).to(device)
    head_vars = torch.zeros((pruning_cfg.n_samples,1)).to(device)

# %%
def save_snapshot(head_losses, head_vars):
    torch.save({"head_losses": head_losses, "head_vars": head_vars}, f"{folder}/{ablation_type}_results.pth")

    plot_no_outliers(
        sns.scatterplot, .03,
        head_losses, head_vars.sqrt(),
        args={"x": "Component mean loss", "y": "Component std loss", "s": 5,
              "f": f"{folder}/{ablation_type}.png"}
    )

# %%
for no_batches in tqdm(range(max_batches)):
    batch, last_token_pos, cf = task_ds.retrieve_batch_cf(tokenizer)

    if oa_train:
        modal_optimizer.zero_grad()

        loss = vertex_pruner(batch, last_token_pos)
        loss.backward()
        modal_optimizer.step()

        if no_batches % -100 == -1:
            torch.save({"modal_attention": vertex_pruner.modal_attention, "modal_mlp": vertex_pruner.modal_mlp}, f"{folder}/{ablation_type}_modes.pth")
            vertex_pruner.log.plot(["kl_loss"], mv=100, save=f"{folder}/{ablation_type}_train.png")
    else:
        with torch.no_grad():
            # loss: [n_components, batch_size]
            loss, _ = vertex_pruner(batch, last_token_pos, counterfactual=cf, separate_loss=True)
            head_losses, head_vars = update_means_variances(head_losses, head_vars, loss, no_batches)
        
        if no_batches % -100 == -1:
            save_snapshot(head_losses, head_vars)

if not oa_train:
    save_snapshot(head_losses, head_vars)