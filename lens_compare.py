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
from utils.training_utils import load_model_data, save_hook_last_token, LinePlot, plot_no_outliers
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

# load in lenses
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

# we need to compute mean activations to analyze projection
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

# std: scalar or shape [n_layers, d_model]
# fixed_dir: "perturb" (local perturbation), "steer" (add a vector), "project" (linear projection out of a direction), False
# if fixed_dir is perturb, then std gives stdev of how much to perturb each neuron
# if fixed_dir is steer, then std gives the direction to add
# if fixed_dir is project, then std gives us the direction to project out of
def causal_and_save_hook_last_token(perturb_type, bsz, std, act_mean, save_to, act, hook):
    act = torch.cat([act, act[:bsz]], dim=0)

    prev_act = act[-bsz:,-1,:].clone()
    if perturb_type == "steer":
        act[-bsz:,-1,:] = act[-bsz:,-1,:] + torch.randn((1,)).to(device) * std
    elif perturb_type == "project":
        orig_act = act[-bsz:,-1,:] - act_mean
        standard_dir = std / std.norm()
        act[-bsz:,-1,:] = act[-bsz:,-1,:] - (orig_act * standard_dir).sum() * standard_dir
    else:
        act[-bsz:,-1,:] = act[-bsz:,-1,:] + torch.randn_like(act[-bsz:,-1,:]).to(device) * std

    save_to.append((prev_act, act[-bsz:,-1,:]))
    return act

# std: [n_layers, d_model]
def run_causal_perturb(batch, std, perturb_type):
    activation_storage = []
    bsz = batch.shape[0]

    target_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[
                    *[(partial(resid_points_filter, layer_no), 
                    partial(causal_and_save_hook_last_token, perturb_type, bsz, std[layer_no], act_means[layer_no], activation_storage)) 
                    for layer_no in range(n_layers)],
                ]
    )[:,-1].softmax(dim=-1)

    target_probs = target_probs.unflatten(0, (n_layers + 1, bsz)).permute((1,0,2))

    no_causal_probs = target_probs[:,[0]]
    target_probs = target_probs[:,1:]

    perturb_loss = kl_loss(target_probs.log(), no_causal_probs).sum(dim=-1)

    return target_probs, no_causal_probs, [a[0] for a in activation_storage], [a[1] for a in activation_storage], perturb_loss

# v1: [b, l, d_vocab], v2: [b, l, d_vocab], orig_dist: [b, 1, d_vocab]
def a_inner_prod(v1, v2, orig_dist):
    assert (orig_dist.sum(dim=-1) - 1).abs().mean() <= 1e-3
    geom_mean_1 = (orig_dist * v1.log()).sum(dim=-1, keepdim=True).exp()
    geom_mean_2 = (orig_dist * v2.log()).sum(dim=-1, keepdim=True).exp()

    return (orig_dist * (v1 / geom_mean_1).log() * (v2 / geom_mean_2).log()).sum(dim=-1)

def a_sim(v1, v2, orig_dist):
    return a_inner_prod(v1, v2, orig_dist) / (a_inner_prod(v1, v1, orig_dist) * (a_inner_prod(v2, v2, orig_dist))).sqrt()

def subtract_probs(v1, v2):
    return (v1.log() - v2.log()).softmax(dim=-1)

# perturb_type: ["fixed", "project", False]
def get_lens_loss(batch, lens_list={"modal", "tuned"}, std=0, perturb_type=False):
    output_losses = {}
    a_sims = {}

    if perturb_type:
        target_probs, orig_probs, orig_acts, activation_storage, output_losses["perturb"] = run_causal_perturb(batch, std, perturb_type)
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

        if perturb_type:
            orig_lens_probs = apply_modal_lens(model, all_lens_weights["modal"], orig_acts, shared_bias)
            output_losses["modal"] = kl_loss(modal_lens_probs.log(), orig_lens_probs).sum(dim=-1)
        else:
            output_losses["modal"] = kl_loss(modal_lens_probs.log(), target_probs).sum(dim=-1)
    
    for k in ["tuned", "lm", "grad"]:
        if k in lens_list:
            lens_probs = apply_lens(model, all_lens_weights[k], all_lens_bias[k], activation_storage)

            if perturb_type:
                orig_lens_probs = apply_lens(model, all_lens_weights[k], all_lens_bias[k], orig_acts)
                output_losses[k] = kl_loss(lens_probs.log(), orig_lens_probs).sum(dim=-1)
                a_sims[k] = a_sim(subtract_probs(lens_probs, orig_lens_probs), 
                                  subtract_probs(target_probs, orig_probs), orig_probs)
            else:
                output_losses[k] = kl_loss(lens_probs.log(), target_probs).sum(dim=-1)

    if perturb_type:
        return output_losses, activation_storage, a_sims
    return output_losses, activation_storage
# %%

def compile_loss_dfs(all_losses, lens_losses_dfs, suffix=""):
    for k in all_losses:
        lens_loss_df = pd.DataFrame(torch.cat(all_losses[k], dim=0).cpu().numpy())
        lens_loss_df.columns = [f"{k}_{x}" for x in lens_loss_df.columns]
        lens_losses_dfs[f"{k}{suffix}"] = lens_loss_df
    return lens_losses_dfs

# plotting tuned vs OCA lens performance
def corr_plot(lens_loss_1, lens_loss_2, key_1, key_2):
    tuned_modal_comp = lens_loss_1.merge(lens_loss_2, left_index=True, right_index=True)

    f, axes = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))
    f, axes_log = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))

    # correlation plot
    for i in range(n_layers):
        plot_no_outliers(
            sns.histplot, .01, 
            tuned_modal_comp[f"{key_1}_{i}"], tuned_modal_comp[f"{key_2}_{i}"], 
            axes[i // 3, i % 3], xy_line=True, 
            args={"x": f"{key_1}_{i}", "y": f"{key_1}_{i}"})
        
        plot_no_outliers(
            sns.histplot, 0,
            np.log(tuned_modal_comp[f"{key_1}_{i}"]), np.log(tuned_modal_comp[f"{key_2}_{i}"]),
            axes_log[i // 3, i % 3], xy_line=True,
            args={"x": f"{key_1}_{i}", "y": f"{key_2}_{i}"}
        )
    plt.show()

def overall_comp(lens_losses_dfs, title="Lens losses", save=None):
    # overall comparison line plot
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


# get vanilla losses
# all_losses = {"modal": [], "tuned": [], "lm": [], "grad": []}
# stds = [torch.zeros(d_model).to(device) for _ in range(n_layers)]
# for i in tqdm(range(100)):
#     batch = next(owt_iter)['tokens']

#     with torch.no_grad():
#         lens_losses, activation_storage = get_lens_loss(batch, list(all_losses.keys()))

#         for k in lens_losses:
#             # [batch, n_layers]
#             all_losses[k].append(torch.nan_to_num(lens_losses[k], nan=0, posinf=0, neginf=0))

#         # compute std for strength of ``local'' causal perturbations
#         for l in range(len(stds)):
#             stds[l] = (i * stds[l] + activation_storage[l].std(dim=0)) / (i+1)
# if not os.path.exists(f"{folders['tuned']}/stds.pkl"):
#     with open(f"{folders['tuned']}/stds.pkl", "wb") as f:
#         pickle.dump(stds, f)

# show vanilla losses
# lens_losses_dfs = {}
# lens_losses_dfs = compile_loss_dfs(all_losses, lens_losses_dfs)
# # corr_plot(lens_losses_dfs["modal"], lens_losses_dfs["tuned"], "modal", "tuned")
# # corr_plot(lens_losses_dfs["modal"], lens_losses_dfs["lm"], "modal", "lm")
# corr_plot(lens_losses_dfs["lm"], lens_losses_dfs["tuned"], "lm", "tuned")
# overall_comp(lens_losses_dfs)

# %%

def get_causal_losses(std, perturb_type, batches=100, lens_list=["modal", "lm", "tuned", "grad", "perturb"]):
    all_causal_losses = {k: [] for k in lens_list}
    all_a_sims = {k: [] for k in lens_list}
    for i in range(batches):
        batch = next(owt_iter)['tokens']

        with torch.no_grad():
            lens_losses, _, a_sims = get_lens_loss(batch, lens_list, std=std, perturb_type=perturb_type)

            for k in lens_losses:
                # [batch, n_layers]
                all_causal_losses[k].append(torch.nan_to_num(lens_losses[k], nan=0, posinf=0, neginf=0))

                if k != "perturb":
                    all_a_sims[k].append(torch.nan_to_num(a_sims[k], nan=0, posinf=0, neginf=0))
    
    for k in all_causal_losses:
        all_causal_losses[k] = torch.cat(all_causal_losses[k], dim=0)

        if k != "perturb":
            all_a_sims[k] = torch.cat(all_a_sims[k], dim=0)
    
    return all_causal_losses, all_a_sims

# %%

with open(f"{folders['tuned']}/stds.pkl", "rb") as f:
    stds = pickle.load(f)
std_intvs = np.logspace(-3, 0.25, 100)

# %%

# For perturb loss, get std threshold for reaching a certain level of KL loss
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
                _, _, _, _, perturb_loss = run_causal_perturb(batch, [std * stds[k] for k in range(n_layers)])
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

# get singular vecs
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

# Experiment 1. importance similarity

# project out of random directions
# project out of directions with highest singular values

# part 1. random directions

# projection losses pass empty object
def get_edit_losses(loss_obj, directions, perturb_type="project", n_batches=1, lens_list=['perturb', 'tuned', 'grad', 'lm']):
    for k in lens_list:
        loss_obj[k] = {"loss": [], "sim": []}
    for vec in tqdm(directions):
        causal_loss, a_sim = get_causal_losses(vec, perturb_type, n_batches, lens_list)
        for lens in lens_list:
            loss_obj[lens]['loss'].append(causal_loss[lens])
            loss_obj[lens]['sim'].append(a_sim[lens])

# random directions
directions = torch.randn((500, n_layers, d_model)).to(device)  
directions = directions / directions.norm(dim=-1, keepdim=True)

# projection
projection_losses = {}
get_edit_losses(projection_losses, directions, n_batches = 1)
torch.save(projection_losses, f"{folders['lm']}/proj_losses_random.pth")

# steering
steering_losses = {}
get_edit_losses(steering_losses, directions, "steer", n_batches = 1)
torch.save(steering_losses, f"{folders['lm']}/steer_losses_random.pth")

# singular vectors projection
for k in ['tuned', 'grad', 'lm']:
    # sing_vecs[k] has singular vecs for this lens for all layers
    directions = sing_vecs[k].permute((2,0,1))[:100]

    projection_losses = {}
    get_edit_losses(projection_losses, directions, n_batches=20, lens_list=[k, 'perturb'])
    torch.save(projection_losses, f"{folders[k]}/proj_losses_singular.pth")

    # steering
    steering_losses = {}
    get_edit_losses(steering_losses, directions, "steer", n_batches=20, lens_list=[k, 'perturb'])
    torch.save(steering_losses, f"{folders[k]}/steer_losses_singular.pth")

# %%

def summarize_loss_obj(proj_losses_file, title):
    proj_losses = torch.load(f"{folders['lm']}/{proj_losses_file}.pth")
    for k in proj_losses:
        proj_losses[k]['loss'] = torch.stack(proj_losses[k]['loss'], dim=0)
    for k in ['tuned','grad', 'lm']:
        proj_losses[k]['sim'] = torch.stack(proj_losses[k]['sim'], dim=0)

    f, axes = plt.subplots(1,3, figsize=(15,5))

    for i,k in enumerate(['tuned','grad','lm']):
        print(k)
        plot_no_outliers(sns.histplot, 0, 
                        proj_losses['perturb']['loss'].log().flatten().cpu(),
                        proj_losses[k]['loss'].log().flatten().cpu(), 
                        axes[i], xy_line=True,
                        args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
    plt.savefig(f"{folders['lm']}/{title}_importance_indiv_points.png")
    plt.show()

    f, axes = plt.subplots(1,3, figsize=(15,5))

    for i,k in enumerate(['tuned','grad','lm']):
        print(k)
        plot_no_outliers(sns.histplot, 0, 
                        proj_losses['perturb']['loss'].mean(dim=1).log().flatten().cpu(),
                        proj_losses[k]['loss'].mean(dim=1).log().flatten().cpu(), 
                        axes[i], xy_line=True,
                        args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
    plt.savefig(f"{folders['lm']}/{title}_importance_directions.png")
    plt.show()

    f, axes = plt.subplots(1,4, figsize=(20,5))

    for i,k in enumerate(['tuned','grad','lm']):
        ax=sns.histplot(proj_losses[k]['sim'].mean(dim=1).flatten().cpu(), ax=axes[i])
        ax.set(**{"xlabel": f"similarity {k}", "ylabel": f"density"})

        sns.lineplot(proj_losses[k]['sim'].mean(dim=[0,1]).flatten().cpu(), ax=axes[-1], label=k)
    plt.savefig(f"{folders['lm']}/{title}_sim_comp.png")
    plt.show()

def summarize_loss_obj_2(proj_losses_file, title):
    proj_losses = {}
    for typ in ['tuned', 'grad', 'lm']:
        proj_losses[typ] = torch.load(f"{folders[typ]}/{proj_losses_file}.pth")
        for k in proj_losses[typ]:
            proj_losses[typ][k]['loss'] = torch.stack(proj_losses[typ][k]['loss'], dim=0)
        proj_losses[typ][typ]['sim'] = torch.stack(proj_losses[typ][typ]['sim'], dim=0)

    f, axes = plt.subplots(1,3, figsize=(15,5))

    for i,k in enumerate(['tuned','grad','lm']):
        plot_no_outliers(sns.histplot, 0, 
                        proj_losses[k]['perturb']['loss'].log().flatten().cpu(),
                        proj_losses[k][k]['loss'].log().flatten().cpu(), 
                        axes[i], xy_line=True,
                        args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
    plt.savefig(f"{folders['lm']}/{title}_importance_indiv_points.png")
    plt.show()

    f, axes = plt.subplots(1,3, figsize=(15,5))

    for i,k in enumerate(['tuned','grad','lm']):
        plot_no_outliers(sns.histplot, 0, 
                        proj_losses[k]['perturb']['loss'].mean(dim=1).log().flatten().cpu(),
                        proj_losses[k][k]['loss'].mean(dim=1).log().flatten().cpu(), 
                        axes[i], xy_line=True,
                        args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
    plt.savefig(f"{folders['lm']}/{title}_importance_directions.png")
    plt.show()

    f, axes = plt.subplots(1,4, figsize=(20,5))

    for i,k in enumerate(['tuned','grad','lm']):
        ax=sns.histplot(proj_losses[k][k]['sim'].mean(dim=1).flatten().cpu(), ax=axes[i])
        ax.set(**{"xlabel": f"similarity {k}", "ylabel": f"density"})

        sns.lineplot(proj_losses[k][k]['sim'].mean(dim=[0,1]).flatten().cpu(), ax=axes[-1], label=k)
    plt.savefig(f"{folders['lm']}/{title}_sim_comp.png")
    plt.show()

# %%
summarize_loss_obj("proj_losses_random", "proj_rand")

# %%
summarize_loss_obj("steer_losses_random", "steer_rand")

# %%
summarize_loss_obj_2("proj_losses_singular", "proj_sing")
summarize_loss_obj_2("steer_losses_singular", "steer_sing")

# %%



# %%
# Experiment 2. stimulus-response

# Degradation of performance

# CBE with resample ablation

# %%

# compile losses for singular vectors
# if os.path.exists(f"{folders['lm']}/causal_losses_lens.pkl"):
#     with open(f"{folders['lm']}/causal_losses_lens.pkl", "rb") as f:
#         causal_losses_lens_feature, causal_losses_model_feature = pickle.load(f)
# else:
#     causal_losses_lens_feature = {}
#     causal_losses_model_feature = {}
#     for k in ["tuned", "grad", "lm"]:
#         causal_losses_lens_feature[k] = {}
#         causal_losses_model_feature[k] = {}

#         for perturb_type in ["perturb", "project"]:

#             causal_losses_lens_feature[k][perturb_type] = []
#             causal_losses_model_feature[k][perturb_type] = []
            
#             for sing_vec_id in range(d_model):
#                 c_l = get_causal_losses(sing_vecs[k][:,:,sing_vec_id], perturb_type, batches=50, lens_list=[k, 'perturb'])
#                 loss_k = torch.cat(c_l[k], dim=0).mean(dim=0)
#                 loss_perturb = torch.cat(c_l['perturb'], dim=0).mean(dim=0)
#                 causal_losses_lens_feature[k][perturb_type].append(loss_k)
#                 causal_losses_model_feature[k][perturb_type].append(loss_perturb)

#             causal_losses_lens_feature[k][perturb_type] = torch.stack(causal_losses_lens_feature[k][perturb_type], dim=0)
#             causal_losses_model_feature[k][perturb_type] = torch.stack(causal_losses_model_feature[k][perturb_type], dim=0)
#             with open(f"{folders['lm']}/causal_losses_lens.pkl", "wb") as f:
#                 pickle.dump((causal_losses_lens_feature, causal_losses_model_feature), f)

# %%

# for fixed_dir in ["perturb", "project"]: 
#     for k in ["tuned", "grad", "lm"]:
#         sns.scatterplot

# casual basis extraction -> projection loss
# atchinson similarity
# Applications: identifying adversarial inputs, difficulty