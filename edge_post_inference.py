# %%
import torch
import os
from sys import argv
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
from functools import partial
import torch.optim
import glob
import pickle
from EdgePruner import EdgePruner
from MaskSampler import ConstantMaskSampler
from MaskConfig import EdgeInferenceConfig
from task_datasets import IOIConfig, GTConfig
from circuit_utils import discretize_mask, prune_dangling_edges, retrieve_mask
from training_utils import load_model_data, LinePlot

# %%
# load model
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# settings
try:
    reg_lamb = float(argv[1])
except:
    reg_lamb=1e-4

folder=f"pruning_edges_auto/gt_edges_unif"
load_edges = False

batch_size=50
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, batch_size=batch_size)
pruning_cfg.lamb = reg_lamb
pruning_cfg.n_samples = 1

task_ds = GTConfig(batch_size, device)
ds_test = task_ds.get_test_set(tokenizer)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

# %%
next_batch = partial(task_ds.next_batch, tokenizer)
pruning_cfg.record_post_training(mask_sampler, edge_pruner, ds_test, next_batch, load_edges=load_edges)
# %%

# %%
