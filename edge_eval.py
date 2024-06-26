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
from pruners.EdgePruner import EdgePruner
from mask_samplers.MaskSampler import ConstantMaskSampler
from utils.MaskConfig import EdgeInferenceConfig
from utils.task_datasets import get_task_ds
from utils.training_utils import load_model_data, LinePlot

# %%

# dataset = "ioi"
# ablation_type = "cf"
dataset = argv[1]
ablation_type = argv[2]

if len(argv) > 3:
    transfer_folder = argv[3]
else:
    transfer_folder = ablation_type
re_eval = True 
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

# %%

load_edges_dict = {
    "acdc": True,
    "eap": True,
    "hc": False, 
    "unif": False
}

folders = []
load_edges = []

for technique in load_edges_dict:
    path = f"results/pruning/{dataset}/{transfer_folder}/{technique}"
    if os.path.exists(path):
        folders.append(path)
        load_edges.append(load_edges_dict[technique])

# %%
batch_size=50
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folders[0], batch_size=batch_size)
pruning_cfg.n_samples = 1

task_ds = get_task_ds(dataset, batch_size, device, ablation_type)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
edge_pruner = EdgePruner(model, pruning_cfg, mask_sampler, **task_ds.get_pruner_args({
    "mean", "mean_agnostic", "resample", "resample_agnostic", "cf", "oa"
}))
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

# %%
next_batch = partial(task_ds.retrieve_batch_cf, tokenizer)
pruning_cfg.record_post_training(folders, edge_pruner, next_batch, ablation_type, load_edges=load_edges, re_eval=re_eval, transfer=(transfer_folder != ablation_type))
# %%
