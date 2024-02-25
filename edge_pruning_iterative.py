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
from edge_pruning_config import IOIConfig, GTConfig
from EdgePruner import EdgePruner, PruneMaskIterativeSampler
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
reg_lamb = float(argv[1])
# reg_lamb=50
gpu_requeue = True

folder=f"pruning_edges_auto/ioi_iter/{reg_lamb}"

if not os.path.exists(folder):
    os.makedirs(folder)

pruning_cfg = IOIConfig(model.cfg, device, folder, 0.5)
pruning_cfg.lamb = reg_lamb
pruning_cfg.record_every = 50
pruning_cfg.temp_avg_intv = 5
pruning_cfg.temp_comp_intv = 20
pruning_cfg.temp_convergence_target = 200

for param in model.parameters():
    param.requires_grad = False

# %%
    
snapshot_path = f"{pruning_cfg.folder}/snapshot.pth"
metadata_path = f"{pruning_cfg.folder}/metadata.pkl"

def take_snapshot(j):
    lp_count.plot(save=f"{pruning_cfg.folder}/train-step{j}.png")

    pruner_dict = edge_pruner.state_dict()
    pruner_dict = {k: pruner_dict[k] for k in pruner_dict if not k.startswith("base_model")}
    torch.save({
        'pruner_dict': pruner_dict,
        'sampling_optim_dict': sampling_optimizer.state_dict(),
        'modal_optim_dict': modal_optimizer.state_dict(),
    }, snapshot_path)

    with open(metadata_path, "wb") as f:
        pickle.dump((edge_pruner.log, lp_count), f)

# %%
mask_sampler = PruneMaskIterativeSampler(pruning_cfg)
edge_pruner = EdgePruner(model, pruning_cfg, mask_sampler, ablation_backward=True)

sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=1e-3)
modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

lp_count = LinePlot(['total_edges', 'step_size', 'mode_step_size'])
# %%

if gpu_requeue and os.path.exists(snapshot_path) and os.path.exists(metadata_path):
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
max_batches = 10000
for no_batches in tqdm(range(edge_pruner.log.t, max_batches)):
    plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
    checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

    batch, last_token_pos = pruning_cfg.next_batch(tokenizer)
    last_token_pos = last_token_pos.int()

    modal_optimizer.zero_grad()
    sampling_optimizer.zero_grad()

    # sample prune mask
    graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
    loss, all_sampling_params = edge_pruner(batch, last_token_pos, graph_suffix)
    loss.backward()

    prev_alphas = all_sampling_params[:,0].detach().clone()
    prev_modes = edge_pruner.get_modes().detach().clone()

    sampling_optimizer.step()
    modal_optimizer.step()

    mask_sampler.fix_nans()

    with torch.no_grad():
        step_sz = (mask_sampler.get_sampling_params()[:,0] - prev_alphas).abs().sum()
        mode_step_sz = (edge_pruner.get_modes().clone() - prev_modes).norm(dim=-1).mean()
        lp_count.add_entry({
            "total_edges": mask_sampler.total_edges - (
                    mask_sampler.get_sampling_params()[...,0] < 0
                ).sum().item(), 
            "step_size": step_sz.item(), 
            "mode_step_size": mode_step_sz.item()
        })

    if plotting:
        take_snapshot("")
        if checkpointing:
            take_snapshot(f"-{no_batches}")            
    
    min_train=100
    if edge_pruner.log.t - edge_pruner.log.last_tick < min_train:
        continue

    edge_decline = edge_pruner.log.stat_sig_growth("complexity_loss", avg_intv=10, comp_intv=40)
    if edge_decline is not False and edge_decline[0] < .03:
        print("DONE WITH", mask_sampler.component_type, mask_sampler.cur_layer)
        mask_sampler.freeze_params()
        if not mask_sampler.next_layer():
            break
        edge_pruner.log.last_tick = edge_pruner.log.t

# %%
