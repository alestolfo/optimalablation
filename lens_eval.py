
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
overall_folder = "results/lens/overall"
all_models = ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def summarize_loss_obj(folders, lens_list, proj_losses_file, title, separate_vecs=False, offset=0, resample_idx=None):
    # lens_list = ['modal', 'tuned', 'grad', 'linear_oa']
    n_layers = None
    
    if separate_vecs:
        proj_losses = {}
        for typ in lens_list:
            if typ == 'modal':
                continue

            proj_losses[typ] = torch.load(f"{folders[typ]}/{proj_losses_file}.pth")
            for k in proj_losses[typ]:
                # modal loss is included in linear_oa
                proj_losses[typ][k]['loss'] = torch.stack(proj_losses[typ][k]['loss'], dim=0)
                if k != "perturb":
                    proj_losses[typ][k]['sim'] = torch.stack(proj_losses[typ][k]['sim'], dim=0)
                
                # shape:
                if n_layers is not None:
                    assert n_layers == proj_losses[typ][k]['loss'].shape[-1]
                else:
                    n_layers = proj_losses[typ][k]['loss'].shape[-1]
    else:
        proj_losses = torch.load(f"{folders['linear_oa']}/{proj_losses_file}.pth")
        for k in proj_losses:
            proj_losses[k]['loss'] = torch.stack(proj_losses[k]['loss'], dim=0)
        for k in lens_list:
            proj_losses[k]['sim'] = torch.stack(proj_losses[k]['sim'], dim=0)
    
        if n_layers is not None:
            # shape: n_dirs, n_samples, n_layers
            assert n_layers == proj_losses[k]['loss'].shape[-1]
        else:
            n_layers = proj_losses[k]['loss'].shape[-1]
    
    print(n_layers)

    num_lens = len(lens_list)
    plot_list = ["points", "dirs", "a_sim"] if resample_idx is None else ['points', 'a_sim']
    figs = {}
    axes = {}
    corrs = {}
    sim_vecs = {}

    for p in plot_list:
        figs[p], axes[p] = plt.subplots(n_layers-offset, num_lens, figsize=(num_lens * 5, (n_layers-offset) * 5))
        figs[p].tight_layout()
        corrs[p] = {}

    for i,k in enumerate(lens_list):
        corrs['points'][k] = []
        sim_vecs[k] = []

        print(k)

        if separate_vecs:
            # info for modal lens is stored inside linear_oa
            dict_key = 'linear_oa' if k == 'modal' else k
            perturb_loss = proj_losses[dict_key]['perturb']['loss']
            lens_result = proj_losses[dict_key][k]
            if resample_idx is None:
                corrs['dirs'][k] = []
            else:
                perturb_loss = perturb_loss[resample_idx]
                lens_result = {'loss': lens_result['loss'][resample_idx], 'sim': lens_result['sim'][resample_idx]}
        else:
            perturb_loss = proj_losses['perturb']['loss']
            lens_result = proj_losses[k]
            corrs['dirs'][k] = []

        for j in range(offset,n_layers):
            model_loss = perturb_loss[...,j]

            corr = plot_no_outliers(sns.histplot, 0, 
                            model_loss.log().flatten().cpu(),
                            lens_result['loss'][...,j].log().flatten().cpu(), 
                            axes['points'][j-offset,i], xy_line=True,
                            args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
            corrs['points'][k].append(corr)

            if resample_idx is None:
                corr = plot_no_outliers(sns.histplot, 0, 
                                model_loss.mean(dim=1).log().flatten().cpu(),
                                lens_result['loss'][...,j].mean(dim=1).log().flatten().cpu(), 
                                axes['dirs'][j-offset,i], xy_line=True,
                                args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
                corrs['dirs'][k].append(corr)

            # large_ls_idx = ((model_loss.flatten() > model_loss.quantile(.5)) * (lens_result['loss'][...,j].flatten() > lens_result['loss'][...,j].quantile(.5))).nonzero()[:,0]
            # sim_vec = lens_result['sim'][...,j].flatten()[large_ls_idx]
            sim_vec = lens_result['sim'][...,j].flatten()
            sim_vecs[k].append(sim_vec.mean().item())

            plot_no_outliers(sns.histplot, 0, 
                            model_loss.log().flatten().cpu(),
                            # model_loss.log().flatten()[large_ls_idx].cpu(),
                            lens_result['sim'][...,j].flatten().cpu(), 
                            # sim_vec.cpu(), 
                            axes['a_sim'][j-offset,i],
                            args={"x": "model loss", "y": f"{k} similarity", "corr": True})

    for p in plot_list:
        figs[p].savefig(f"{folders['linear_oa']}/{title}_importance_{p}.png")
        plt.close(figs[p])
    
    torch.save({"corrs": corrs, "sim_vecs": sim_vecs}, f"{folders['linear_oa']}/causal_plot_{title}.pth")

# %%

# all_models = ["gpt2-xl"]
for model_name in all_models:
    folders = {
        "modal": f"results/lens/{model_name}/oa",
        "linear_oa": f"results/lens/{model_name}/linear_oa",
        "tuned": f"results/lens/{model_name}/tuned",
        "grad": f"results/lens/{model_name}/grad"
    }

    print(model_name)

    # grad lens not included for larger models
    if model_name == "gpt2-small":
        lens_list = ["tuned", "grad", "modal", "linear_oa"]
    else:
        lens_list = ["tuned", "modal", "linear_oa"]

    summarize_loss_obj(folders, lens_list, "proj_losses_random", "proj_rand")
    summarize_loss_obj(folders, lens_list, "steer_losses_random", "steer_rand")
    summarize_loss_obj(folders, lens_list, "proj_losses_singular", "proj_sing", True)
    summarize_loss_obj(folders, lens_list, "steer_losses_singular", "steer_sing", True)
    summarize_loss_obj(folders, lens_list, "proj_losses_singular", "proj_sing", True)
    summarize_loss_obj(folders, lens_list, "steer_losses_singular", "steer_sing", True)

    for i, resample_ct in enumerate([5,10,20,50,100]):
        summarize_loss_obj(folders, lens_list, "resample_losses_singular", f"resample_{resample_ct}", True, resample_idx=i)

# %%

ax_labels={"tuned": "Tuned lens", "modal": "OCA lens", "mean": "mean ablation", "resample": "resample ablation"}

def load_plots(folders, lens_list, title, resample=False, offset=0, loop=False, lw=None, prev_fig=None, prev_axes=None):

    n_plots = 3
    f, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots,5))

    data = torch.load(f"{folders['linear_oa']}/causal_plot_{title}.pth")
    corrs = data['corrs']
    sim_vecs = data['sim_vecs']

    for i, k in enumerate(lens_list):
        n_layers = len(corrs['points'][k])
        x = np.arange(n_layers-offset) + offset

        ax = sns.lineplot(x=x, y=corrs['points'][k], ax=axes[0], label=k, linewidth=lw)
        axes[0].set(xlabel="Layer no", ylabel="Correlation")

        ax = sns.lineplot(x=x, y=corrs['dirs'][k], ax=axes[1], label=k)
        axes[1].set(label="Layer no", ylabel="Correlation")

        ax = sns.lineplot(x=x, y=sim_vecs[k], ax=axes[-1], label=k, linewidth=lw)
        axes[-1].set(xlabel="Layer no", ylabel="Similarity")
    
    if loop:
        return f, axes

    m, n = title.split("_")
    m = {"proj": "projection", "steer": "steering"}[m]
    n = {"rand": "with random directions", "sing": "with singular vectors"}[n]

    plt.suptitle(f"Causal faithfulness: {m} {n}")
    plt.tight_layout()
    f.savefig(f"{folders['linear_oa']}/summary_{title}.png")
    f.show()
    plt.close(f)

# %%
title = "steer_sing"

offset = 0
lens_list = ["tuned", "modal"]
blue_shades = plt.cm.Greens(np.linspace( 0.3, 1, 4))
red_shades = plt.cm.Reds(np.linspace( 0.3, 1, 4))

n_plots = 3
f, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

# model_name = "gpt2-medium"
# all_titles = ["proj_rand", "proj_sing"]
for j, model_name in enumerate(all_models):
# for j, title in enumerate(all_titles):
    folders = {
        "modal": f"results/lens/{model_name}/oa",
        "linear_oa": f"results/lens/{model_name}/linear_oa",
        "tuned": f"results/lens/{model_name}/tuned",
        "grad": f"results/lens/{model_name}/grad"
    }

    data = torch.load(f"{folders['linear_oa']}/causal_plot_{title}.pth")
    corrs = data['corrs']
    sim_vecs = data['sim_vecs']

    for i, k in enumerate(lens_list):
        if k == "tuned":
            shade = red_shades[j]
            ls = None
        else:
            shade = red_shades[j]
            ls = "dashed"
        
        n_layers = len(corrs['points'][k])
        x = np.arange(n_layers-offset) + offset

        ax = sns.lineplot(x=x, y=corrs['points'][k], ax=axes[0], label=f"{model_name}, {ax_labels[k]}", color=shade, linestyle=ls)
        axes[0].set(xlabel="Layer number", ylabel="Correlation")

        ax = sns.lineplot(x=x, y=corrs['dirs'][k], ax=axes[1], label=f"{model_name}, {ax_labels[k]}", color=shade, linestyle=ls)
        axes[1].set(xlabel="Layer number", ylabel="Correlation")

        ax = sns.lineplot(x=x, y=sim_vecs[k], ax=axes[-1], label=f"{model_name}, {ax_labels[k]}", color=shade, linestyle=ls)
        axes[-1].set(xlabel="Layer number", ylabel="Similarity")
    
    # m, n = title.split("_")
    # m = {"proj": "projection", "steer": "steering"}[m]
    # n = {"rand": "with random directions", "sing": "with singular vectors"}[n]

titles = {"proj_sing": "basis projection", "proj_rand": "random projection", "steer_rand": "random perturbation", "steer_sing": "basis-aligned purturbation"}
plt.suptitle(f"Causal faithfulness: {titles[title]}", fontsize=24)
plt.tight_layout()
f.savefig(f"{overall_folder}/summary_{title}.png")
plt.show()
plt.close(f)

    # load_plots(folders, lens_list, "proj_rand")
    # load_plots(folders, lens_list, "steer_rand")
    # load_plots(folders, lens_list, "proj_sing")
    # load_plots(folders, lens_list, "steer_sing")
    # f, axes = None, None
    # for i, resample_ct in enumerate([5,10,20,50,100]):
    #     f, axes = load_plots(folders, lens_list, f"resample_{resample_ct}", resample=True, loop=True, prev_fig=f, prev_axes=axes)
    # plt.suptitle(f"Causal faithfulness: resample ablation loss")
    # plt.tight_layout()
    # f.savefig(f"{folders['linear_oa']}/summary_resample.png")
    # f.show()
    # plt.close(f)
    # break

# %%
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels

colors = {"modal": "orange", "tuned": "blue"}
for model_name in all_models:
    folders = {
        "modal": f"results/lens/{model_name}/oa",
        "linear_oa": f"results/lens/{model_name}/linear_oa",
        "tuned": f"results/lens/{model_name}/tuned",
        "grad": f"results/lens/{model_name}/grad"
    }

    n_plots = 2
    f, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

    blue_shades = plt.cm.Greens(np.linspace( 0.5, 1, 5))
    red_shades = plt.cm.Reds(np.linspace( 0.3, 0.8, 5))

    all_data = []
    for n_dirs in [5, 10, 20, 50, 100]:
        data = torch.load(f"{folders['linear_oa']}/causal_plot_resample_{n_dirs}.pth")
        all_data.append((n_dirs, data))

    for i, k in enumerate(lens_list):
        if k == "tuned":
            shades = blue_shades
        else:
            shades = red_shades 

        for j, (n_dirs, data) in enumerate(all_data):
            corrs = data['corrs']
            sim_vecs = data['sim_vecs']
            n_layers = len(corrs['points'][k])
            x = np.arange(n_layers)

            ax = sns.lineplot(x=x, y=corrs['points'][k], ax=axes[0], label=f"{ax_labels[k]} ({n_dirs})", linewidth=math.log(n_dirs) / 2, color=shades[j], linestyle="dashed")
            axes[0].set(xlabel="Layer number", ylabel="Correlation")

            ax = sns.lineplot(x=x, y=sim_vecs[k], ax=axes[-1], label=f"{ax_labels[k]} ({n_dirs})", linewidth=math.log(n_dirs) / 2, color=shades[j], linestyle="dashed")
            axes[-1].set(xlabel="Layer number", ylabel="Similarity")

    plt.suptitle(f"Causal faithfulness: resample basis directions, {model_name}", fontsize=24)
    plt.tight_layout()
    f.savefig(f"{overall_folder}/{model_name}_summary_resample.png")
    plt.show()
    plt.close(f)
    

# %%

def get_shades(lens):
    green_shades = plt.cm.Greens(np.linspace( 0.8, 0.3, 8))
    blue_shades = plt.cm.Blues(np.linspace( 0.8, 0.3,8))
    red_shades = plt.cm.Reds(np.linspace( 0.8, 0.3, 8))
    yellow_shades = plt.cm.YlOrBr(np.linspace( 0.8, 0.3, 8))
    if lens == "modal":
        shades = red_shades
    elif lens == "tuned":
        shades = green_shades
    elif lens == "mean":
        shades = blue_shades
    else:
        shades = yellow_shades
    return shades


for model_name in all_models:
    print(model_name)
    folders = {
        "modal": f"results/lens/{model_name}/oa",
        "linear_oa": f"results/lens/{model_name}/linear_oa",
        "tuned": f"results/lens/{model_name}/tuned",
        "grad": f"results/lens/{model_name}/grad"
    }
    print(model_name)
    mn = model_name
    lw = [2,1.5, 1,0.5,0.3,0.2,0.1]
    series = {}
    f = plt.subplots(figsize=(5,5))
    vanilla_losses = torch.load(f"{folders['linear_oa']}/original.pth")
    lens_list = ["mean", "resample", "modal", "tuned"]
    for k in vanilla_losses:
        if k not in lens_list:
            continue
        # print(k)
        # print(list(vanilla_losses[f"{k}"]))
        sns.lineplot(list(vanilla_losses[f"{k}"]), color=get_shades(k)[0], label=f"{ax_labels[k].capitalize()}", linewidth=lw[i])
        # plt.show()
        plt.xlabel("Layer number", fontsize=16)
        plt.ylabel("KL-divergence", fontsize=16)
        plt.title(f"Lens losses on {mn}", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"results/lens/overall-{mn}.png")
    continue
    causal_losses = torch.load(f"{folders['linear_oa']}/causal_losses.pth")
    for i, t in enumerate(causal_losses):
        for lens in causal_losses[t][2]:
            if lens not in lens_list:
                continue
            if len(causal_losses[t][2][lens]) == 0:
                continue
            shade = get_shades(lens)
            loss_curve = causal_losses[t][2][lens].mean(dim=0).cpu().numpy()
            sns.lineplot(loss_curve, color=shade[i+1], linestyle="dashed", label=f"{ax_labels[lens]} ({t})", linewidth=lw[i])
    plt.show()
    

    # if not os.path.exists(f"{folders['linear_oa']}/original.p"):
    #     vanilla_losses = exp.get_vanilla_losses(lens_list, pics_folder=folders['linear_oa'])
    #     torch.save(vanilla_losses, f"{folders['linear_oa']}/original.pth")

    # # get causal perturb losses
    # if not os.path.exists(f"{folders['linear_oa']}/causal_losses.p"):
    #     exp.get_causal_perturb_losses(lens_list, save=f"{folders['linear_oa']}/causal_losses.pth", pics_folder=folders['linear_oa'])

# %%
