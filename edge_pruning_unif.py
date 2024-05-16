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
from mask_samplers.EdgeMaskSampler import EdgeMaskUnifSampler
from utils.MaskConfig import EdgeInferenceConfig
from utils.task_datasets import get_task_ds
from utils.training_utils import load_model_data, LinePlot, load_args
from pruners.EdgePruner import EdgePruner
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
args = load_args("pruning_edges_auto", 1.8e-4, {"name": "edges_unif"})
folder, reg_lamb, dataset = args["folder"], args["lamb"], args["dataset"]
node_reg = min(0.2 * reg_lamb * 146, 1e-2)

# if dataset == "gt":
#     node_reg = 5e-4
# else:
#     node_reg = 5e-3

gpu_requeue = True
# reset_optim = 1000

pretrained_folder = None
# f"pruning_edges_auto/ioi/300.0"

pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, init_param=1)
pruning_cfg.lamb = reg_lamb

task_ds = get_task_ds(dataset, pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%

# prune_retrain = True
# prune_length = 200
# retrain_length = 100

# %%
mask_sampler = EdgeMaskUnifSampler(pruning_cfg, node_reg=node_reg)
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)

take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)
# %%
pruning_cfg.record_every = 50

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

    grad_norms = mask_sampler.clip_grad(5)

    prev_alphas = mask_sampler.get_sampling_params()[:,0].detach().clone()
    prev_modes = edge_pruner.get_modes().detach().clone()

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

        # bernoulli comparison plot
        # grad = mask_sampler.sampling_params['attn-attn'][9].grad.flatten()
        # print(grad[grad.nonzero()].shape)
        # sns.scatterplot(x=mask_sampler.sampled_mask['attn-attn'][9].float().mean(dim=0).flatten()[grad.nonzero()].flatten().cpu().detach(),y=grad[grad.nonzero()].flatten().cpu())
        # plt.xlabel("Prob inclusion in batch")
        # plt.ylabel("Autograd")
        # plt.savefig(f"bernoulli/prior/unif_prob_grad_{j}.png")
        # plt.close()

        # sns.scatterplot(x=mask_sampler.sampling_params['attn-attn'][9].float().detach().flatten()[grad.nonzero()].flatten().cpu().detach(),y=grad[grad.nonzero()].flatten().cpu())
        # plt.xlabel("Sampling parameter")
        # plt.ylabel("Autograd")
        # plt.savefig(f"bernoulli/prior/unif_param_grad_{j}.png")

        if checkpointing:
            take_snapshot(f"-{no_batches}")
# %%
