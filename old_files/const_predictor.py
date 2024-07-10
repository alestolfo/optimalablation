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
from utils.training_utils import load_model_data, LinePlot, load_args, plot_no_outliers
from pruners.EdgePruner import EdgePruner
# %%

dataset = "ioi"

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
d_model = model.cfg.d_model

# %%
batch_size = 100
task_ds = get_task_ds(dataset, batch_size, device)

last_resid_hook = f"blocks.{n_layers-1}.hook_resid_post"

def cache_hook(activation_storage, last_token_pos, act, hook):
    activation_storage.append(
        act[
            torch.arange(act.shape[0]).to(device),
            last_token_pos
        ])

# %%

with torch.no_grad():
    activation_storage = []
    for i in tqdm(range(200)):
        batch, last_token_pos, cf = task_ds.retrieve_batch_cf(tokenizer)
        result = model.run_with_hooks(
            batch,
            fwd_hooks=[(last_resid_hook, partial(cache_hook, activation_storage, last_token_pos))]
        )

init_constant = torch.cat(activation_storage, dim=0).mean(dim=0)
# %%
kl_loss = torch.nn.KLDivLoss(reduction="none")

lr=1e-3
const_prediction = torch.nn.Parameter(init_constant)
optimizer = torch.optim.AdamW([const_prediction], lr=lr, weight_decay=0)

for i in tqdm(range(10000)):
    batch, last_token_pos, cf = task_ds.retrieve_batch_cf(tokenizer)
    with torch.no_grad():
        result = model(batch)
        result = result[
            torch.arange(batch_size).to(device),
            last_token_pos
        ].log_softmax(dim=-1)
    
    pred_result = const_prediction[None, None, :].expand(batch_size, -1, -1)
    pred_result = model.ln_final(pred_result)
    pred_result = model.unembed(pred_result)
    pred_result = pred_result[:,0].log_softmax(dim=-1)

    loss = kl_loss(pred_result, result.exp()).sum(dim=-1)
    loss.mean().backward()

    optimizer.step()

    print("mean KL", loss.mean())

# %%
