# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
import datasets
from itertools import cycle
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
from sys import argv
import math
from functools import partial
import torch.optim
import time
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, LinePlot
from task_datasets import OWTConfig, IOIConfig, GTConfig
# %%
# import sys
# del sys.modules['task_datasets']
# %%
# dataset settings

folder = "oca/gt"

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 100
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
# model.cfg.use_attn_result = True

# %%
# task_ds = OWTConfig(owt_iter, device)
# task_ds = IOIConfig(batch_size, device)
task_ds = GTConfig(batch_size, device)

# %%
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model

kl_loss = torch.nn.KLDivLoss(reduction="none")

attn_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out"
mlp_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"
resid_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_post"

# %%

def compute_mean_hook(last_token_pos, activation_storage, activations, hook):
    if not isinstance(last_token_pos, torch.Tensor):
        last_token_pos = last_token_pos * torch.ones(activations.shape[0],).to(device)
    indic_sample = (torch.arange(activations.shape[1]).repeat(activations.shape[0],1).to(device) <= last_token_pos.unsqueeze(1))
    while len(activations.shape) > len(indic_sample.shape):
        indic_sample = indic_sample.unsqueeze(-1)
    reprs = activations * indic_sample
    early_pos = reprs[:,:9].sum(dim=0) / indic_sample[:,:9].sum(dim=0)
    late_pos = (reprs[:,9:].sum(dim=[0,1]) / indic_sample[:,9:].sum(dim=[0,1])).unsqueeze(0)
    activation_storage.append(torch.cat([early_pos,late_pos],dim=0))

def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def mean_ablation_hook_layer_all_tokens(constants, bsz, activations, hook):
    constants = torch.cat([constants[:9], constants[9:]], dim=0)
    activations[-bsz,1:] = constants[1:]
    return activations

def resample_ablation_hook_layer_all_tokens(bsz, activations, hook):
    activations[-bsz,1:] = activations[-bsz + torch.randperm(bsz).to(device),1:]
    return activations

def mode_ablation_hook_layer_all_tokens(constants, bsz, activations, hook):
    activations[-bsz,1:] = constants
    return activations

# %%

attn_means = None
mlp_means = None

for i in tqdm(range(1000)):
    # modify depending on the dataset

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    
    with torch.no_grad():
        attn_storage = []
        mlp_storage = []
        fwd_hooks = [
            *[(partial(attn_post_filter, layer_no), 
                    partial(compute_mean_hook,
                            last_token_pos,
                            attn_storage)
                        ) for layer_no in range(n_layers)],
            *[(partial(mlp_post_filter, layer_no), 
                    partial(compute_mean_hook,
                            last_token_pos,
                            mlp_storage)
                        ) for layer_no in range(n_layers)]
        ]

        model_results = model.run_with_hooks(
                batch,
                fwd_hooks=fwd_hooks
        )
        if attn_means is None:
            attn_means = torch.stack(attn_storage, dim=0)
            mlp_means = torch.stack(mlp_storage, dim=0)
        else:
            attn_means = (i * attn_means + torch.stack(attn_storage, dim=0)) / (i+1)
            mlp_means = (i * mlp_means + torch.stack(mlp_storage, dim=0)) / (i+1)

# %%
with open(f"{folder}/attn_layer_means.pkl", "wb") as f:
    pickle.dump(attn_means,f)
with open(f"{folder}/mlp_layer_means.pkl", "wb") as f:
    pickle.dump(mlp_means,f)

# %%
