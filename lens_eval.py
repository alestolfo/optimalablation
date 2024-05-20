
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

# all_models = ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]
all_models = ["gpt2-xl"]
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


def load_plots(folders, lens_list, title, resample=False, offset=0):

    n_plots = 2 if resample else 3
    f, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots,5))

    data = torch.load(f"{folders['linear_oa']}/causal_plot_{title}.pth")
    corrs = data['corrs']
    sim_vecs = data['sim_vecs']

    for i, k in enumerate(lens_list):
        n_layers = len(corrs['points'][k])
        x = np.arange(n_layers-offset) + offset

        sns.lineplot(x=x, y=corrs['points'][k], ax=axes[0], label=k)

        if not resample:
            sns.lineplot(x=x, y=corrs['dirs'][k], ax=axes[1], label=k)
        # ax=sns.histplot(proj_losses[k]['sim'].mean(dim=1).flatten().cpu(), ax=axes[i])
        # ax.set(**{"xlabel": f"similarity {k}", "ylabel": f"density"})

        # sns.lineplot(lens_result['sim'].mean(dim=[0,1]).flatten().cpu(), ax=axes[-1], label=k)
        sns.lineplot(x=x, y=sim_vecs[k], ax=axes[-1], label=k)


    f.savefig(f"{folders['linear_oa']}/summary_{title}.png")
    f.show()

# %%
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
        lens_list = ["tuned", "modal"]
    else:
        lens_list = ["tuned", "modal"]

    load_plots(folders, lens_list, "proj_rand")
    load_plots(folders, lens_list, "steer_rand")
    load_plots(folders, lens_list, "proj_sing")
    load_plots(folders, lens_list, "steer_sing")
    load_plots(folders, lens_list, "proj_sing")
    load_plots(folders, lens_list, "steer_sing")

    for i, resample_ct in enumerate([5,10,20,50,100]):
        load_plots(folders, lens_list, f"resample_{resample_ct}", resample=True)

# %%
# Experiment 2. stimulus-response

# Degradation of performance

# CBE with resample ablation

# %%
