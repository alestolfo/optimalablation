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
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from EdgePruner import EdgePruner, PruneMaskJointSampler
from edge_pruning_config import IOIConfig, GTConfig
from training_utils import load_model_data, LinePlot

# %%
# load model
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

# %%

# settings
# reg_lamb = float(argv[1])
reg_lamb=500.

folder=f"pruning_edges_auto/ioi/{reg_lamb}"

pruning_cfg = IOIConfig(model.cfg, device, folder)
pruning_cfg.lamb = reg_lamb

for param in model.parameters():
    param.requires_grad = False

# %%
    
snapshot_path = f"{pruning_cfg.folder}/snapshot.pth"
metadata_path = f"{pruning_cfg.folder}/metadata.pkl"

# %%
if os.path.exists(snapshot_path) and os.path.exists(metadata_path):
    print("Loading previous training run")
    previous_state = torch.load(snapshot_path)
    print(previous_state['pruner_dict'].keys())

    modal_attention = previous_state['pruner_dict']['modal_attention']
    modal_mlp = previous_state['pruner_dict']['modal_mlp']
    prune_mask = {}
    for k in previous_state['pruner_dict']:
        if not k.startswith("mask_sampler"):
            continue
        s = k.split(".")
        if s[-2] not in prune_mask:
            prune_mask[s[-2]] = []
        prune_mask[s[-2]].append(previous_state['pruner_dict'][k][...,0])
        if int(s[-1])+1 != len(prune_mask[s[-2]]):
            print("WARNING: out of order")

# %%
all_alphas = torch.cat([ts.flatten() for k in prune_mask for ts in prune_mask[k]], dim=0)
sorted_values, _ = torch.sort(all_alphas)
sns.histplot(sorted_values.cpu())

# %%

constant_prune_mask = {k: [(ts > 0) * 1 for ts in prune_mask[k]] for k in prune_mask}
num_edges = np.sum([np.sum([torch.sum(ts).item() for ts in constant_prune_mask[k]]) for k in constant_prune_mask])
print("num edges", num_edges)

# %%
# attn-attn: (bsz * n_samples) x n_heads (dest) x i x n_heads (source)
# mlp-attn: (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x 1 (d_model)

# attn-mlp: (bsz * n_samples) x i x n_heads
# mlp-mlp: (bsz * n_samples) x 1 (seq_pos) x i x 1 (d_model)

# check for dangling edges
attn_edges_out = torch.zeros((n_layers, n_heads)).to(device)
mlp_edges_out = torch.zeros((n_layers + 2)).to(device)
mlp_edges_out[-1] = 1

for component_type, cur_layer in reversed(pruning_cfg.layers_to_prune):
    if component_type == "mlp":
        this_layer_mlp = mlp_edges_out[cur_layer+1] > 0
        constant_prune_mask['attn-mlp'][cur_layer] *= this_layer_mlp
        attn_edges_out[:min(cur_layer+1, n_layers)] += constant_prune_mask['attn-mlp'][cur_layer] * this_layer_mlp

        constant_prune_mask['mlp-mlp'][cur_layer] *= this_layer_mlp
        mlp_edges_out[:cur_layer+1] += constant_prune_mask['mlp-mlp'][cur_layer] * this_layer_mlp

    if component_type == "attn":
        this_layer_heads = (attn_edges_out[cur_layer] > 0).unsqueeze(-1)
        constant_prune_mask['attn-attn'][cur_layer] *= this_layer_heads.unsqueeze(-1)
        attn_edges_out[:cur_layer] += (constant_prune_mask['attn-attn'][cur_layer] * this_layer_heads.unsqueeze(-1)).sum(dim=[0,1])

        constant_prune_mask['mlp-attn'][cur_layer] *= this_layer_heads
        mlp_edges_out[:cur_layer+1] += (constant_prune_mask['mlp-attn'][cur_layer] * this_layer_heads).sum(dim=[0,1])

attn_edges_in = torch.zeros((n_layers, n_heads)).to(device)
mlp_edges_in = torch.zeros((n_layers + 2)).to(device)
mlp_edges_in[0] = 1

for component_type, cur_layer in pruning_cfg.layers_to_prune:
    if component_type == "mlp":
        constant_prune_mask['attn-mlp'][cur_layer] *= (attn_edges_in[:min(cur_layer+1, n_layers)] > 0)
        mlp_edges_in[cur_layer+1] += constant_prune_mask['attn-mlp'][cur_layer].sum()

        constant_prune_mask['mlp-mlp'][cur_layer] *= (mlp_edges_in[:cur_layer+1] > 0)     
        mlp_edges_in[cur_layer+1] += constant_prune_mask['mlp-mlp'][cur_layer].sum()

    if component_type == "attn":
        constant_prune_mask['attn-attn'][cur_layer] *= (attn_edges_in[:cur_layer] > 0)
        attn_edges_in[cur_layer] += constant_prune_mask['attn-attn'][cur_layer].sum(dim=[0,-2,-1])

        constant_prune_mask['mlp-attn'][cur_layer] *= (mlp_edges_in[:cur_layer+1] > 0)
        attn_edges_in[cur_layer] += constant_prune_mask['mlp-attn'][cur_layer].sum(dim=[0,-1])

num_edges = np.sum([np.sum([torch.sum(ts).item() for ts in constant_prune_mask[k]]) for k in constant_prune_mask])
print("num edges", num_edges)


constant_prune_mask = {k: [ts.unsqueeze(0) for ts in prune_mask[k]] for k in prune_mask}

def sample_constant_mask(constant_prune_mask):
    return constant_prune_mask, None


# %%
edge_pruner = EdgePruner(model, pruning_cfg, partial(sample_constant_mask, constant_prune_mask), inference_mode=True)

# %%

kl_losses = []
for b in pruning_cfg.ds_test:
    batch, last_token_pos = pruning_cfg.next_batch(tokenizer, b)

    with torch.no_grad():
        loss, all_sampling_params = edge_pruner(batch, last_token_pos)
    # kl_losses.append(loss)

# %%
        
# %%
mask_sampler = PruneMaskJointSampler(pruning_cfg)
edge_pruner = EdgePruner(model, pruning_cfg, mask_sampler)

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=1e-3)
modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

lp_count = LinePlot(['step_size', 'mode_step_size'])
# %%

if os.path.exists(snapshot_path) and os.path.exists(metadata_path):
    print("Loading previous training run")
    previous_state = torch.load(snapshot_path)
    edge_pruner.load_state_dict(previous_state['pruner_dict'], strict=False)
    sampling_optimizer.load_state_dict(previous_state['sampling_optim_dict'])
    modal_optimizer.load_state_dict(previous_state['modal_optim_dict'])

    with open(metadata_path, "rb") as f:
        main_log, lp_count = pickle.load(f)
    edge_pruner.set_log(main_log)
else:
    print("New training run")


# %%

max_batches = 3000
for b in pruning_cfg.ds_test:

    # plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
    # checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

    batch, last_token_pos = pruning_cfg.next_batch(tokenizer, b)
    last_token_pos = last_token_pos.int()

    modal_optimizer.zero_grad()
    sampling_optimizer.zero_grad()

    # sample prune mask
    # graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
    loss, all_sampling_params = edge_pruner(batch, last_token_pos)
    loss.backward()

    prev_alphas = all_sampling_params[:,0].detach().clone()
    prev_modes = edge_pruner.get_modes().detach().clone()

    sampling_optimizer.step()
    modal_optimizer.step()

    mask_sampler.fix_nans()

    with torch.no_grad():
        step_sz = (mask_sampler.get_sampling_params()[:,0] - prev_alphas).abs().sum()
        mode_step_sz = (edge_pruner.get_modes().clone() - prev_modes).norm(dim=-1).mean()
        lp_count.add_entry({"step_size": step_sz.item(), "mode_step_size": mode_step_sz.item()})

    # if plotting:
    #     take_snapshot("")
    #     if checkpointing:
    #         take_snapshot(f"-{no_batches}")
    #     if edge_pruner.early_term() >= 10:
    #         take_snapshot("-final")
    #         break


# %%
