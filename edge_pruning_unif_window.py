# %%
import torch
import datasets
import os
import numpy as np
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
from mask_samplers.EdgeMaskSampler import EdgeMaskUnifWindowSampler
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
args = load_args("pruning_edges_auto", 1.9e-4, {"name": "edges_unifsd"})
folder, reg_lamb, dataset = args["folder"], args["lamb"], args["dataset"]
node_reg=2e-3
gpu_requeue = True
# reset_optim = 1000

pretrained_folder = None
# f"pruning_edges_auto/ioi/300.0"

pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, init_param=0)
pruning_cfg.lamb = reg_lamb
pruning_cfg.initialize_params_probs(1)

task_ds = get_task_ds(dataset, pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = EdgeMaskUnifWindowSampler(pruning_cfg, node_reg=node_reg)
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)

take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)

# %%
temp_settings = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]

def compute_bias(plot_folder, max_batches=30):
    mask_sampler.fix_mask = True

    # bias estimation
    bias_avgs = []

    for temp_setting in temp_settings:
        edge_pruner.temp_scale = temp_setting

        all_grads = []
        grad_diffs = []
        for no_batches in tqdm(range(max_batches)):
            batch, last_token_pos = task_ds.next_batch(tokenizer)
            last_token_pos = last_token_pos.int()

            modal_optimizer.zero_grad()
            sampling_optimizer.zero_grad()

            # sample prune mask
            mask_sampler.sample_mask_single_edge()
            loss = edge_pruner(batch, last_token_pos, None)
            loss.backward()

            grads = mask_sampler.get_sampling_grads()
            all_grads.append(grads[mask_sampler.selected_edges])

            modal_optimizer.zero_grad()
            sampling_optimizer.zero_grad()

            mask_sampler.sample_mask_only_edge()
            loss = edge_pruner(batch, last_token_pos, None)
            loss.backward()

            grad_diff = grads - mask_sampler.get_sampling_grads()
            grad_diffs.append(grad_diff[mask_sampler.selected_edges])

            grad_norms = mask_sampler.clip_grad(5)

        all_grads = torch.cat(all_grads, dim=0)
        grad_diffs = torch.cat(grad_diffs, dim=0)

        cur_ax = sns.histplot(x=all_grads.log().flatten().cpu(), y=grad_diffs.log().flatten().cpu())
        min_val = max(cur_ax.get_xlim()[0],cur_ax.get_ylim()[0])
        max_val = min(cur_ax.get_xlim()[1],cur_ax.get_ylim()[1])
        cur_ax.plot([min_val, max_val],[min_val, max_val], color="red", linestyle="-")
        plt.savefig(f"{plot_folder}/grad_bias_{temp_setting}.png")
        plt.xlabel("Grad")
        plt.ylabel("Grad bias")
        plt.title(f"Scale {temp_setting}")
        plt.close()

        cur_ax = sns.histplot(x=all_grads.log().flatten().cpu(), y=(grad_diffs / all_grads).flatten().cpu().clip(min=-5,max=5))
        plt.savefig(f"{plot_folder}/grad_bias_prop_{temp_setting}.png")
        plt.xlabel("Grad")
        plt.ylabel("Grad bias pct")
        plt.title(f"Scale {temp_setting}")
        plt.close()

        bias_avgs.append(grad_diff.mean().item())

    mask_sampler.fix_mask = False

    return bias_avgs
# %%

# variance estimation
def compute_variance(plot_folder, max_batches=30):
    variance_avgs = []

    for temp_setting in temp_settings:
        mask_sampler.temp_scale = temp_setting

        for no_batches in tqdm(range(max_batches)):
            batch, last_token_pos = task_ds.next_batch(tokenizer)
            last_token_pos = last_token_pos.int()

            modal_optimizer.zero_grad()
            sampling_optimizer.zero_grad()

            # sample prune mask
            loss = edge_pruner(batch, last_token_pos, None)
            loss.backward()

            mask_sampler.compile_grads()

            grad_norms = mask_sampler.clip_grad(5)

            mask_sampler.fix_nans()
        
        sns.histplot(x=mask_sampler.running_mean.log().flatten().cpu(), y=mask_sampler.running_variance.log().flatten().cpu())
        plt.savefig(f"{plot_folder}/vars_{temp_setting}.png")
        plt.xlabel("Means")
        plt.ylabel("Variances")
        plt.title(f"Scale {temp_setting}")
        plt.close()

        sns.histplot(x=mask_sampler.running_samples.flatten().cpu(), y=mask_sampler.running_variance.log().flatten().cpu())
        plt.savefig(f"{plot_folder}/vars_samples_{temp_setting}.png")
        plt.xlabel("Samples")
        plt.ylabel("Variances")
        plt.title(f"Scale {temp_setting}")
        plt.close()

        print(mask_sampler.running_variance)
        print((mask_sampler.running_variance < 0).sum())
        print(mask_sampler.running_samples)

        variance_avgs.append(torch.where(
                mask_sampler.running_samples > 0,
                mask_sampler.running_variance / mask_sampler.running_samples,
                0
            ).mean().item())

        print(variance_avgs)
        
        mask_sampler.reset_grads()

    return variance_avgs

# %%

pruning_cfg.record_every = 500
bias_variance_checkpoints = [1, 50, 100, 200, 400, 800, 1500, 2500, 4000]

max_batches = 4001
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
        if checkpointing:
            take_snapshot(f"-{no_batches}")
    
    if no_batches in bias_variance_checkpoints:
        edge_pruner.pause_log = True

        folder = f"results/unif_window/{no_batches}"

        if not os.path.exists(folder):
            os.makedirs(folder)
        bias_avgs = compute_bias(folder)
        variance_avgs = compute_variance(folder)

        sns.lineplot(x=temp_settings, y=bias_avgs, label="Bias")
        sns.lineplot(x=temp_settings, y=variance_avgs, label="Variance")
        sns.lineplot(x=temp_settings, y=np.square(np.array(bias_avgs)) + np.array(variance_avgs), label="Squared Error")
        plt.legend()
        plt.title("MSE comparison")
        plt.savefig(f"{folder}/temp_comparison.png")
        plt.close()

        edge_pruner.pause_log = False
        mask_sampler.temp_scale = 1
  # %%
