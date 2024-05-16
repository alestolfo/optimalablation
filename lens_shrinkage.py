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
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, LinePlot, plot_no_outliers
from utils.lens_utils import LensExperiment, compile_loss_dfs, corr_plot, overall_comp

# %%
sns.set()

folders = {
    "modal": "results/lens/oa",
    "linear_oa": "results/lens/linear_oa",
    "tuned": "results/lens/tuned",
    "grad": "results/lens/grad",
    "shrinkage": "results/lens/oa_shrinkage"
}
shared_bias = False
n_layers = 12

# %%
model_name = "gpt2-small"
batch_size = 20
# 100K OWT samples with default sequence length: 235134
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

# %%

exp = LensExperiment(model, owt_iter, folders, device)

# %%
tuned_wts = exp.all_lens_weights['tuned']
linear_oa_wts = exp.all_lens_weights['linear_oa']

tuned_wts = torch.stack(tuned_wts, dim=0)
linear_oa_wts = torch.stack(linear_oa_wts, dim=0)

tuned_bias = torch.stack(exp.all_lens_bias['tuned'], dim=0)
linear_oa_bias = torch.stack(exp.all_lens_bias['linear_oa'], dim=0)
act_means = torch.stack(exp.act_means, dim=0)
# %%
def linreg(y, x):
    y = y.flatten(start_dim=1, end_dim=-1)
    x = x.flatten(start_dim=1, end_dim=-1)
    return einsum("n_layers wt, n_layers wt -> n_layers", y, x) / einsum("n_layers wt, n_layers wt -> n_layers", x, x)

# %%
n_estimators = 12

def apply_shrinkage(shrinkage_coeff, baseline_wts, baseline_bias):
    shrinkage_wt = torch.tensor(np.linspace(0, 1, n_estimators, endpoint=False), dtype=act_means.dtype).to(device)
    shrinkage_estimators = shrinkage_coeff + (1 - shrinkage_coeff) * shrinkage_wt[:, None]

    for i, est in enumerate(shrinkage_estimators):
        shrank_weights = est[:, None, None] * baseline_wts
        shrank_bias = baseline_bias + einsum("n_layers d_output d_model, n_layers d_model -> n_layers d_output", baseline_wts - shrank_weights, act_means)

        exp.all_lens_weights[f"shrinkage_{i}/{n_estimators}"] = shrank_weights
        exp.all_lens_bias[f"shrinkage_{i}/{n_estimators}"] = shrank_bias

# %%
# shrink tuned lens
shrinkage_coeff = linreg(linear_oa_wts, tuned_wts)
shrinkage_coeff = torch.zeros_like(shrinkage_coeff, device=device)
apply_shrinkage(shrinkage_coeff, tuned_wts, tuned_bias)

# %%

# shrink linear_oa lens
shrinkage_coeff = linreg(tuned_wts, linear_oa_wts)
shrinkage_coeff = torch.zeros_like(shrinkage_coeff, device=device)
apply_shrinkage(shrinkage_coeff, linear_oa_wts, linear_oa_bias)

# %%
lens_list = ["linear_oa", "tuned", *[f"shrinkage_{i}/{n_estimators}" for i in range(n_estimators)]]
# get vanilla losses
if not os.path.exists(f"{folders['shrinkage']}/original.png"):
    exp.get_vanilla_losses(lens_list, pics_folder=folders['shrinkage'])

# %%
# get causal perturb losses
if not os.path.exists(f"{folders['shrinkage']}/causal_losses_z.pth"):
    exp.get_causal_perturb_losses(lens_list, save=f"{folders['shrinkage']}/causal_losses_z.pth", pics_folder=folders['shrinkage'])


# %%


# %%

# %%
print(linreg(tuned_wts, linear_oa_wts))

# %%
print(linreg(linear_oa_wts, tuned_wts))



# %%

reg_coef
# %%

sns.regplot(x=tuned_wts[11].detach().flatten().cpu(), y=oa_wts[11].detach().flatten().cpu())

# %%
