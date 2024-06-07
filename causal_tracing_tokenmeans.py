# %%
import torch
import json
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from glob import glob
from functools import partial
import os
import torch.optim
import time
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, LinePlot
from torch.utils.data import DataLoader
from utils.tracing_utils import get_subject_tokens

# %%

# filter for correct prompts

sns.set()

ds_path = "utils/datasets/facts"
folder="results/ct-new"

# "attribute", "fact"
mode = "fact"
ds_name = "my_facts" if mode == "fact" else "my_attributes"
# %%
model_name = "gpt2-xl"
batch_size = 5
clip_value = 1e5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)
tokenizer = model.tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-2

# learning hyperparameters
kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%
with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
    ds = pickle.load(f)

train_split = 0.6
train_split = math.floor(0.6 * len(ds))
data_loader = DataLoader(ds[:train_split], batch_size=batch_size, shuffle=True)
data_iter = iter(data_loader)

# %%

def save_subject_token(subject_token_pos, storage, act, hook):
    storage.append(act[subject_token_pos[:,0], subject_token_pos[:,1]])

# %%
lp = LinePlot(["kl_loss", "step_size"])
activation_storage = []
for batch in tqdm(data_iter):
    tokens, subject_token_pos = get_subject_tokens(batch, tokenizer)
    
    with torch.no_grad():
        model.run_with_hooks(
            tokens, 
            fwd_hooks=[
                ("hook_embed", partial(save_subject_token, subject_token_pos, activation_storage))
            ]
        )

subject_means = torch.cat(activation_storage, dim=0).mean(dim=0)
torch.save(subject_means, f"{folder}/{mode}/subject_means.pth")
# %%
