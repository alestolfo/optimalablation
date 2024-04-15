# %%
import torch
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
import math
from sys import argv
import os
from functools import partial
import torch.optim
import time
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.MaskConfig import EdgeInferenceConfig
from task_datasets import IOIConfig, GTConfig
from EdgePruner import EdgePruner
from mask_samplers.MaskSampler import EdgeMaskIterativeSampler
from training_utils import load_model_data, LinePlot

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

# relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

# %%

# settings
try:
    reg_lamb = float(argv[1])
except:
    reg_lamb=3e-3
gpu_requeue = True

folder=f"pruning_edges_auto/ioi_iter/{reg_lamb}"

if not os.path.exists(folder):
    os.makedirs(folder)

pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, init_param=0)
pruning_cfg.lamb = reg_lamb
pruning_cfg.record_every = 50
pruning_cfg.temp_avg_intv = 5
pruning_cfg.temp_comp_intv = 20
pruning_cfg.temp_convergence_target = 200

task_ds = IOIConfig(pruning_cfg.batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = EdgeMaskIterativeSampler(pruning_cfg)
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, ablation_backward=True)
edge_pruner.log.pref_start = 0
edge_pruner.add_cache_hooks()
edge_pruner.base_model.add_hook(*edge_pruner.patching_hooks["patch_final"])

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0)
modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)
lp_count.pref_start = 0

if lp_count.stat_list[-1] != "total_edges":
    lp_count.stat_list.append("total_edges")
    lp_count.stat_book["total_edges"] = []

take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)

# %%
max_batches = 20000
finish_count = 0
print(edge_pruner.log.t)
for no_batches in tqdm(range(edge_pruner.log.t, max_batches)):
    plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
    checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    last_token_pos = last_token_pos.int()

    modal_optimizer.zero_grad()
    sampling_optimizer.zero_grad()

    # sample prune mask
    graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
    loss = edge_pruner(batch, last_token_pos, graph_suffix, timing=False)
    loss.mean().backward()

    prev_alphas = mask_sampler.get_sampling_params()[:,0].detach().clone()
    prev_modes = edge_pruner.get_modes().detach().clone()

    sampling_optimizer.step()
    modal_optimizer.step()

    mask_sampler.fix_nans()

    with torch.no_grad():
        step_sz = (mask_sampler.get_sampling_params()[:,0] - prev_alphas).abs()
        step_sz = (step_sz - 1e-3).relu().sum() / (step_sz > 1e-3).sum()
        mode_step_sz = (edge_pruner.get_modes().clone() - prev_modes).norm(dim=-1).mean()
        edges_ablated = (mask_sampler.get_sampling_params()[...,0] < -1).sum().item()
        edges_kept = (mask_sampler.get_sampling_params()[...,0] >= -1).sum().item()
        lp_count.add_entry({
            "total_edges": mask_sampler.total_edges - edges_ablated, 
            "step_size": step_sz.item(), 
            "mode_step_size": mode_step_sz.item()
        })

    if plotting:
        take_snapshot("")
        if checkpointing:
            take_snapshot(f"-{no_batches}")      

    min_train=40

    if edge_pruner.log.t - edge_pruner.log.last_tick < min_train:
        continue

    edge_decline = edge_pruner.log.stat_sig_growth("complexity_loss", avg_intv=6, comp_intv=30)
    if edge_decline is not False and edge_decline[1] < 0 and (edge_decline[0] < .03 or edge_decline[0] * edges_kept < 5) and edge_pruner.log.stat_book['temp'][-1] < 1e-2:
        finish_count += 1
        if edges_kept <= 10 or finish_count >= 20:
            finish_count = 0
            print("DONE WITH", mask_sampler.component_type, mask_sampler.cur_layer)
            pruning_cfg.reset_temp()
            mask_sampler.freeze_params()
            if not mask_sampler.next_layer():
                break
            component_type, cur_layer = mask_sampler.layers_to_prune[mask_sampler.layer_idx]
            if component_type == "attn":
                for circ in edge_pruner.circs:
                    hook_name = f"patch_attn_{cur_layer}_{circ}"
                    edge_pruner.base_model.add_hook(*edge_pruner.patching_hooks[hook_name])
            if component_type == "mlp":
                hook_name = f"patch_mlp_{cur_layer}"
                edge_pruner.base_model.add_hook(*edge_pruner.patching_hooks[hook_name])
            print((mask_sampler.get_sampling_params()[...,0] >= 0).sum().item())
            edge_pruner.log.last_tick = edge_pruner.log.t
    else:
        finish_count = 0

# %%

# prune_mask = {k : [((ts[...,0] > 0) * 1).unsqueeze(0) for ts in mask_sampler.sampling_params[k]] for k in mask_sampler.sampling_params}
# with open('pruning_edges_auto/ioi_iter_round_1/0.001/mask-status.pkl', "wb") as f:
#     pickle.dump((4, prune_mask), f)
# %%
