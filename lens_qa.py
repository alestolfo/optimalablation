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
model_name = "gpt2-small"
dir_mode = "vanilla"

folders = {
    "modal": f"results/lens/{model_name}/oa",
    "linear_oa": f"results/lens/{model_name}/linear_oa",
    "tuned": f"results/lens/{model_name}/tuned",
    "grad": f"results/lens/{model_name}/grad",
    "mean": f"results/lens/{model_name}/mean",
    "resample": f"results/lens/{model_name}/resample"
}

# %%

batch_size = 20
# 100K OWT samples with default sequence length: 235134
device, model, tokenizer, owt_iter = load_model_data("gpt2-small", batch_size)
model = HookedTransformer.from_pretrained(model_name, device=device)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

# %%

from utils.datasets.lens.dataset_params import *
from utils.datasets.lens.prompt_params import *
from utils.datasets.lens.demo_params import *
from utils.datasets.lens.model_params import *
from utils.lens_datasets import Prefixes, get_dataset

# %%

dataset_name = 'sick'

dataset_params = DATASET_PARAMS[dataset_name]
prompt_params = PROMPT_PARAMS[dataset_name]
demo_params = DEMO_PARAMS["permuted_incorrect_labels"]
num_inputs = 1000

model_list = {
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2_medium",
    "gpt2-large": "gpt2_large",
    "gpt2-xl": "gpt2_xl",
}
model_params = MODEL_PARAMS[model_list[model_name]]
n_demos = model_params['max_demos']

for i, prompt_params_i in prompt_params.items():
    prefixes = Prefixes(
        get_dataset(dataset_params),
        prompt_params_i,
        demo_params,
        model_params,
        tokenizer,
        num_inputs,
        n_demos,
    )



# %%
for k in folders:
    if not os.path.exists(folders[k]):
        os.makedirs(folders[k])

print(model_name)
print(dir_mode)

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

# Experiments: resample top singular vecs + [projection, steering] * [random, singular]

# random vectors
N_RAND_DIRS = 2000
EXAMPLES_RAND = 25

# singular vectors
EXAMPLES_SINGULAR = 200

# resample
EXAMPLES_RESAMPLE = 1000

# %%
# model_name = "gpt2-small"
batch_size = CAUSAL_BATCH_SIZE
# 100K OWT samples with default sequence length: 235134
device, model, tokenizer, owt_iter = load_model_data("gpt2-small", batch_size)
model = HookedTransformer.from_pretrained(model_name, device=device)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

shared_bias = False
# n_layers = 12

BATCHES_RAND = (EXAMPLES_RAND - 1) // CAUSAL_BATCH_SIZE + 1
BATCHES_SINGULAR = (EXAMPLES_SINGULAR - 1) // CAUSAL_BATCH_SIZE + 1
BATCHES_RESAMPLE = (EXAMPLES_RESAMPLE - 1) // CAUSAL_BATCH_SIZE + 1

print(BATCHES_RAND, BATCHES_SINGULAR, BATCHES_RESAMPLE)

# when i don't want to load the model
# n_layers = 12

# %%

exp = LensExperiment(model, owt_iter, folders, device)
# lens_list = ["modal", "linear_oa", "tuned", "grad"]

# %%
# grad lens not included for larger models
if model_name == "gpt2-small":
    lens_list = ["modal", "linear_oa", "tuned", "grad"]
else:
    lens_list = ["modal", "linear_oa", "tuned"]

# %%
if dir_mode == "vanilla":
    if "mean" not in lens_list:
        lens_list.append("mean")
    
    if "resample" not in lens_list:
        lens_list.append("resample")
    
    # get vanilla losses
    if not os.path.exists(f"{folders['linear_oa']}/original.p"):
        vanilla_losses = exp.get_vanilla_losses(lens_list, pics_folder=folders['linear_oa'])
        torch.save(vanilla_losses, f"{folders['linear_oa']}/original.pth")

    # get causal perturb losses
    # if not os.path.exists(f"{folders['linear_oa']}/causal_losses.p"):
    #     exp.get_causal_perturb_losses(lens_list, save=f"{folders['linear_oa']}/causal_losses.pth", pics_folder=folders['linear_oa'])

    exit()

# %%

# get singular vecs
# right singular vectors are the columns of v
sing_vecs = {}
sing_vals = {}
for k in exp.all_lens_weights:
    if k == "modal":
        continue

    sing_vecs[k] = []
    sing_vals[k] = []

    for j,ts in enumerate(exp.all_lens_weights[k]):
        # v: [d_norm n_vecs]
        u, s, v = torch.linalg.svd(ts @ exp.a_mtrx[j])
        # a_mtrx: [d_mvn d_norm]
        vecs = exp.a_mtrx[j] @ v
        vecs = vecs / vecs.norm(dim=0)
        sing_vecs[k].append(v)
        sing_vals[k].append(s)
    
    sing_vecs[k] = torch.stack(sing_vecs[k], dim=0)
    sing_vals[k] = torch.stack(sing_vals[k], dim=0)

# %%
for k in sing_vecs:
    # test orthogonality of columms: W^TW should be diagonal
    dot_products = einsum("n_layers d_model n_vecs, n_layers d_model n_vecs_2 -> n_layers n_vecs n_vecs_2", sing_vecs[k], sing_vecs[k])
    for vecs_layer in dot_products:
        diff_eye = vecs_layer - torch.eye(d_model).to(device)
        # weird?
        # assert diff_eye.diag().abs().mean() < 5e-4
        # assert diff_eye.abs().mean() < 1e-4
    # print(sing_vecs[k].shape)
