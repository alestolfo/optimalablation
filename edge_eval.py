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
from utils.circuit_utils import discretize_mask, prune_dangling_edges, retrieve_mask
from utils.training_utils import load_model_data, LinePlot

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
dataset = "ioi"
ablation_type="cf"
folders=[
    f"results/pruning/{dataset}/{ablation_type}/acdc",
    # f"results/pruning/{dataset}/{ablation_type}/eap",
    f"results/pruning/{dataset}/{ablation_type}/hc",
    f"results/pruning/{dataset}/{ablation_type}/unif",
]
load_edges = [
    True,
    # True,
    False,
    False
]
cf_mode = ablation_type in {"resample", "cf"}

batch_size=50
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folders[0], batch_size=batch_size)
pruning_cfg.n_samples = 1

task_ds = get_task_ds(dataset, batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, counterfactual_mode=cf_mode)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

# %%
next_batch = partial(task_ds.retrieve_batch_cf, tokenizer, ablation_type, test=True)
pruning_cfg.record_post_training(folders, edge_pruner, next_batch, ablation_type, load_edges=load_edges)
# %%
