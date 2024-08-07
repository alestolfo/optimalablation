# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
import os
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
from sys import argv
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, LinePlot, plot_no_outliers
from utils.lens_utils import LensExperiment, compile_loss_dfs, corr_plot, overall_comp

# %%
sns.set()

# model_name = argv[1]
# dir_mode = argv[2]
model_name = "gpt2-large"
# dir_mode = "vanilla"
print(model_name)
# print(dir_mode)

folders = {
    "modal": f"results/lens/{model_name}/oa",
    "linear_oa": f"results/lens/{model_name}/linear_oa",
    "tuned": f"results/lens/{model_name}/tuned",
    "grad": f"results/lens/{model_name}/grad",
    "mean": f"results/lens/{model_name}/mean",
    "resample": f"results/lens/{model_name}/resample"
}

for k in folders:
    if not os.path.exists(folders[k]):
        os.makedirs(folders[k])


if model_name == "gpt2-xl":
    CAUSAL_BATCH_SIZE = 3
elif model_name == "gpt2-large":
    CAUSAL_BATCH_SIZE = 7
elif model_name == "gpt2-medium":
    CAUSAL_BATCH_SIZE = 12
elif model_name == "gpt2-small":
    CAUSAL_BATCH_SIZE = 25
else:
    raise Exception("Model not found")

# %%
# model_name = "gpt2-small"
batch_size = CAUSAL_BATCH_SIZE * 30
# 100K OWT samples with default sequence length: 235134
device="cuda:0"
model = HookedTransformer.from_pretrained(model_name, device=device)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

shared_bias = False

# when i don't want to load the model
# n_layers = 12

# %%

exp = LensExperiment(model, None, folders, device)

# %%
# grad lens not included for larger models
lens_list = ["modal", "tuned"]

# %%
for ctx_length in [5,10,15,35,50,75,100]:
    batch_factor = min(25 / ctx_length, 1)
    custom_bsz = math.ceil(batch_size * batch_factor)
    device, model, tokenizer, owt_iter = load_model_data("gpt2-small", custom_bsz, ctx_length=ctx_length)
    exp.owt_iter = owt_iter
    if not os.path.exists(f"{folders['linear_oa']}/original_general_{ctx_length}.pth"):
        vanilla_losses = exp.get_vanilla_losses(lens_list, pics_folder=folders['linear_oa'], no_batches=math.ceil(100 / batch_factor))
        torch.save(vanilla_losses, f"{folders['linear_oa']}/orig_general_{ctx_length}.pth")

# %%
