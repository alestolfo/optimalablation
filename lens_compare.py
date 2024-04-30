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
from utils.training_utils import load_model_data, save_hook_last_token, LinePlot
from utils.lens_utils import apply_lens, apply_modal_lens

# %%
sns.set()

folders = {
    "modal": "results/modal_lens/random_init",
    "lm": "results/modal_lens/linear_oca",
    "tuned": "results/tuned_lens",
    "grad": "results/modal_lens/grad_baseline"
}
shared_bias = False

# %%
model_name = "gpt2-small"
batch_size = 20
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

all_lens_weights = {}
all_lens_bias = {}

for k in folders:
    if os.path.exists(f"{folders[k]}/lens_weights.pkl"):
        with open(f"{folders[k]}/lens_weights.pkl", "rb") as f:
            all_lens_weights[k] = pickle.load(f)
    if os.path.exists(f"{folders[k]}/lens_bias.pkl"):
        with open(f"{folders[k]}/lens_bias.pkl", "rb") as f:
            all_lens_bias[k] = pickle.load(f)

# %%

if os.path.exists(f"{folders['lm']}/act_means.pth"):
    act_means = torch.load(f"{folders['lm']}/act_means.pth")
else:
    # compute act means
    act_means = [0 for _ in range(n_layers)]
    for i in tqdm(range(1000)):
        batch = next(owt_iter)['tokens']

        with torch.no_grad():
            activation_storage = []

            target_probs = model.run_with_hooks(
                batch,
                fwd_hooks=[
                        *[(partial(resid_points_filter, layer_no), 
                        partial(save_hook_last_token, activation_storage)) 
                        for layer_no in range(n_layers)],
                    ]
            )[:,-1].softmax(dim=-1).unsqueeze(1)

            for l in range(len(act_means)):
                act_means[l] = (i * act_means[l] + activation_storage[l].mean(dim=0)) / (i+1)

    act_means = torch.stack(act_means, dim=0)
    torch.save(act_means, f"{folders['lm']}/act_means.pth")
# %%

# std: scalar or shape [d_model,]
# fixed_dir: "perturb", "project", False
def causal_and_save_hook_last_token(fixed_dir, bsz, std, act_mean, save_to, act, hook):
    act = torch.cat([act, act[:bsz]], dim=0)

    prev_act = act[-bsz:,-1,:].clone()
    if fixed_dir == "perturb":
        act[-bsz:,-1,:] = act[-bsz:,-1,:] + torch.randn((1,)).to(device) * std
    elif fixed_dir == "project":
        orig_act = act[-bsz:,-1,:] - act_mean
        standard_dir = std / std.norm()
        act[-bsz:,-1,:] = orig_act - (orig_act * standard_dir).sum() * standard_dir
    else:
        act[-bsz:,-1,:] = act[-bsz:,-1,:] + torch.randn_like(act[-bsz:,-1,:]).to(device) * std

    save_to.append((prev_act, act[-bsz:,-1,:]))
    return act

def run_causal_perturb(batch, std, fixed_dir):
    activation_storage = []
    bsz = batch.shape[0]

    target_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[
                    *[(partial(resid_points_filter, layer_no), 
                    partial(causal_and_save_hook_last_token, fixed_dir, bsz, std[layer_no], act_means[layer_no], activation_storage)) 
                    for layer_no in range(n_layers)],
                ]
    )[:,-1].softmax(dim=-1)

    target_probs = target_probs.unflatten(0, (n_layers + 1, bsz)).permute((1,0,2))

    no_causal_probs = target_probs[:,[0]]
    target_probs = target_probs[:,1:]

    perturb_loss = kl_loss(target_probs.log(), no_causal_probs).sum(dim=-1)

    return target_probs, [a[0] for a in activation_storage], [a[1] for a in activation_storage], perturb_loss

# fixed_dir: "fixed", "project", False
def get_lens_loss(batch, lens_list={"modal", "tuned"}, causal=False, std=0, fixed_dir=False):
    output_losses = {}

    if causal:
        target_probs, orig_acts, activation_storage, output_losses["perturb"] = run_causal_perturb(batch, std, fixed_dir)
    else:
        activation_storage = []

        target_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[
                    *[(partial(resid_points_filter, layer_no), 
                    partial(save_hook_last_token, activation_storage)) 
                    for layer_no in range(n_layers)],
                ]
        )[:,-1].softmax(dim=-1).unsqueeze(1)

    if "modal" in lens_list:
        modal_lens_probs = apply_modal_lens(model, all_lens_weights["modal"], activation_storage, shared_bias)
        output_losses["modal"] = kl_loss(modal_lens_probs.log(), target_probs).sum(dim=-1)
    
    for k in ["tuned", "lm", "grad"]:
        if k in lens_list:
            lens_probs = apply_lens(model, all_lens_weights[k], all_lens_bias[k], activation_storage)

            if causal and fixed_dir:
                orig_lens_probs = apply_lens(model, all_lens_weights[k], all_lens_bias[k], orig_acts)
                output_losses[k] = kl_loss(lens_probs.log(), orig_lens_probs).sum(dim=-1)
            else:
                output_losses[k] = kl_loss(lens_probs.log(), target_probs).sum(dim=-1)

    return output_losses, activation_storage

# %%

all_losses = {"modal": [], "tuned": [], "lm": [], "grad": []}
stds = [torch.zeros(d_model).to(device) for _ in range(n_layers)]
for i in tqdm(range(100)):
    batch = next(owt_iter)['tokens']

    with torch.no_grad():
        lens_losses, activation_storage = get_lens_loss(batch, list(all_losses.keys()))

        for k in lens_losses:
            # [batch, n_layers]
            all_losses[k].append(torch.nan_to_num(lens_losses[k], nan=0, posinf=0, neginf=0))

        for l in range(len(stds)):
            stds[l] = (i * stds[l] + activation_storage[l].std(dim=0)) / (i+1)
if not os.path.exists(f"{folders['tuned']}/stds.pkl"):
    with open(f"{folders['tuned']}/stds.pkl", "wb") as f:
        pickle.dump(stds, f)

# %%

def compile_loss_dfs(all_losses, lens_losses_dfs, suffix=""):
    for k in all_losses:
        lens_loss_df = pd.DataFrame(torch.cat(all_losses[k], dim=0).cpu().numpy())
        lens_loss_df.columns = [f"{k}_{x}" for x in lens_loss_df.columns]
        lens_losses_dfs[f"{k}{suffix}"] = lens_loss_df
    return lens_losses_dfs

def corr_plot(lens_loss_1, lens_loss_2, key_1, key_2):
    tuned_modal_comp = lens_loss_1.merge(lens_loss_2, left_index=True, right_index=True)

    f, axes = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))
    f, axes_log = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))

    # correlation plot
    for i in range(n_layers):
        cur_ax = sns.histplot(x=(tuned_modal_comp[f"{key_1}_{i}"]), y=(tuned_modal_comp[f"{key_2}_{i}"]), ax=axes[i // 3, i % 3])
        cur_ax.set_xlim(tuned_modal_comp[f"{key_1}_{i}"].quantile(.01), tuned_modal_comp[f"{key_1}_{i}"].quantile(.99))
        cur_ax.set_ylim(tuned_modal_comp[f"{key_2}_{i}"].quantile(.01), tuned_modal_comp[f"{key_2}_{i}"].quantile(.99))
        cur_ax.set(xlabel=f"{key_1}_{i}", ylabel=f"{key_2}_{i}")
        min_val = max(cur_ax.get_xlim()[0],cur_ax.get_ylim()[0])
        max_val = min(cur_ax.get_xlim()[1],cur_ax.get_ylim()[1])
        cur_ax.plot([min_val, max_val],[min_val, max_val], color="red", linestyle="-")

        cur_ax = sns.histplot(x=np.log(tuned_modal_comp[f"{key_1}_{i}"]), y=np.log(tuned_modal_comp[f"{key_2}_{i}"]), ax=axes_log[i // 3, i % 3])
        min_val = max(cur_ax.get_xlim()[0],cur_ax.get_ylim()[0])
        max_val = min(cur_ax.get_xlim()[1],cur_ax.get_ylim()[1])
        cur_ax.plot([min_val, max_val],[min_val, max_val], color="red", linestyle="-")
        cur_ax.set(xlabel=f"{key_1}_{i}", ylabel=f"{key_2}_{i}")
    plt.show()

def overall_comp(lens_losses_dfs, title="Lens losses", save=None):
    # overall comparison
    lens_loss_means = {}
    for k in lens_losses_dfs:
        lens_loss_means[k] = lens_losses_dfs[k].mean()
        ax = sns.lineplot(x=[i for i in range(len(lens_loss_means[k]))], y=lens_loss_means[k], label=k)
        ax.set(xlabel="layer", ylabel="KL-divergence", title=title)

    if save:
        plt.savefig(save)
    plt.show()
    plt.close()

# %%

# show vanilla losses
lens_losses_dfs = {}
lens_losses_dfs = compile_loss_dfs(all_losses, lens_losses_dfs)
# corr_plot(lens_losses_dfs["modal"], lens_losses_dfs["tuned"], "modal", "tuned")
# corr_plot(lens_losses_dfs["modal"], lens_losses_dfs["lm"], "modal", "lm")
corr_plot(lens_losses_dfs["lm"], lens_losses_dfs["tuned"], "lm", "tuned")
overall_comp(lens_losses_dfs)

# %%

def get_causal_losses(std, fixed_dir, batches=100, lens_list=["modal", "lm", "tuned", "grad", "perturb"]):
    all_causal_losses = {k: [] for k in lens_list}
    for i in tqdm(range(batches)):
        batch = next(owt_iter)['tokens']

        with torch.no_grad():
            lens_losses, activation_storage = get_lens_loss(batch, lens_list, causal=True, std=std, fixed_dir=fixed_dir)

            for k in lens_losses:
                # [batch, n_layers]
                all_causal_losses[k].append(torch.nan_to_num(lens_losses[k], nan=0, posinf=0, neginf=0))
    
    return all_causal_losses

# %%

with open(f"{folders['tuned']}/stds.pkl", "rb") as f:
    stds = pickle.load(f)
std_intvs = np.logspace(-3, 0.25, 100)

# %%

# get std threshold

if os.path.exists(f"{folders['tuned']}/perturb_losses.pkl"):
    with open(f"{folders['tuned']}/perturb_losses.pkl", "rb") as f:
        perturb_losses = pickle.load(f)
else:
    perturb_losses = []

    with torch.no_grad():
        for std in std_intvs:
            perturb_losses_by_std = []
            for i in tqdm(range(50)):
                batch = next(owt_iter)['tokens']
                _, _, perturb_loss = run_causal_perturb(batch, [std * stds[k] for k in range(n_layers)])
                perturb_losses_by_std.append(perturb_loss.mean(dim=0))
            perturb_losses_by_std = torch.stack(perturb_losses_by_std, dim=0).mean(dim=0)
            perturb_losses.append(perturb_losses_by_std)

    with open(f"{folders['tuned']}/perturb_losses.pkl", "wb") as f:
        pickle.dump(perturb_losses, f)
    for layer_no in range(n_layers):
        sns.lineplot(x=std_intvs, y=[(ts[layer_no].item()) for ts in perturb_losses], label=layer_no)
perturb_losses = torch.stack(perturb_losses, dim=0)

# %%
# kl_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 1]

# causal_losses = {}
# for t in kl_thresholds:
#     above_t = (torch.arange(perturb_losses.shape[0]).unsqueeze(1).repeat(1,perturb_losses.shape[1]).to(device) * (perturb_losses < t)).argmax(dim=0)
#     causal_magnitudes = torch.tensor(std_intvs).to(device)[above_t].unsqueeze(-1) * torch.stack(stds, dim=0).to(device)
#     causal_losses[t] = get_causal_losses(causal_magnitudes, False)
# # %%

# with open(f"{folders['tuned']}/causal_losses.pkl", "wb") as f:
#     pickle.dump(causal_losses, f)

# # %%

# for t in causal_losses: 
#     causal_lens_losses_dfs = {}
#     causal_lens_losses_dfs = compile_loss_dfs(causal_losses[t], causal_lens_losses_dfs, suffix="_causal")
#     print(causal_lens_losses_dfs.keys())
#     causal_lens_losses_dfs = {**causal_lens_losses_dfs, **lens_losses_dfs}
#     overall_comp(causal_lens_losses_dfs, title=f"Lens losses causal {t}", save=f"{folders['lm']}/causal_{t}.png")
# %%

# right singular vectors are the columns of v
sing_vecs = {}
sing_vals = {}
for k in all_lens_weights:
    if k == "modal":
        continue

    sing_vecs[k] = []
    sing_vals[k] = []

    for ts in all_lens_weights[k]:
        u, s, v = torch.linalg.svd(ts)
        sing_vecs[k].append(v)
        sing_vals[k].append(s)
    
    sing_vecs[k] = torch.stack(sing_vecs[k], dim=0)
    sing_vals[k] = torch.stack(sing_vals[k], dim=0)

# %%

if os.path.exists(f"{folders['lm']}/causal_losses_lens.pkl"):
    with open(f"{folders['lm']}/causal_losses_lens.pkl", "wb") as f:
        causal_losses_lens_feature, causal_losses_model_feature = pickle.load(f)
else:
    causal_losses_lens_feature = {}
    causal_losses_model_feature = {}
    for k in ["tuned", "grad", "lm"]:

        causal_losses_lens_feature[k] = {}
        causal_losses_model_feature[k] = {}

        for fixed_dir in ["perturb", "project"]:

            causal_losses_lens_feature[k][fixed_dir] = []
            causal_losses_model_feature[k][fixed_dir] = []
            
            for sing_vec_id in range(d_model):
                c_l = get_causal_losses(sing_vecs[k][:,:,sing_vec_id], fixed_dir, batches=50, lens_list=[k, 'perturb'])
                loss_k = torch.cat(c_l[k], dim=0).mean(dim=0)
                loss_perturb = torch.cat(c_l['perturb'], dim=0).mean(dim=0)
                causal_losses_lens_feature[k][fixed_dir].append(loss_k)
                causal_losses_model_feature[k][fixed_dir].append(loss_perturb)

            causal_losses_lens_feature[k][fixed_dir] = torch.stack(causal_losses_lens_feature[k][fixed_dir], dim=0)
            causal_losses_model_feature[k][fixed_dir] = torch.stack(causal_losses_model_feature[k][fixed_dir], dim=0)
            with open(f"{folders['lm']}/causal_losses_lens.pkl", "wb") as f:
                pickle.dump((causal_losses_lens_feature, causal_losses_model_feature), f)

# %%

# for fixed_dir in ["perturb", "project"]: 
#     for k in ["tuned", "grad", "lm"]:
#         sns.scatterplot

# casual basis extraction -> projection loss
# atchinson similarity
# Applications: identifying adversarial inputs, difficulty