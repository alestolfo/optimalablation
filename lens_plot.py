

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


CORR_SIZE = 20
SMALL_SIZE = 24
MEDIUM_SIZE = 24
BIGGER_SIZE = 32

plt.rc('font', size=CORR_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=CORR_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plot_folder = "plots_export/lens"

if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

all_models = ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]

ax_labels={"tuned": "Tuned lens", "modal": "OCA lens", "mean": "Mean", "resample": "Resample"}
# %%
titles = {"proj_sing": "basis projection", "proj_rand": "random projection", "steer_rand": "random perturbation", "steer_sing": "basis-aligned perturbation", "resample_100": "resample basis directions"}

for title in titles:
    offset = 0
    lens_list = ["tuned", "modal"]
    blue_shades = plt.cm.Greens(np.linspace( 0.3, 1, 4))
    red_shades = plt.cm.Reds(np.linspace( 0.3, 1, 4))

    n_plots = 2
    f, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots + 3, 6))

    # plt.rc('axes', titlesize=12)     # fontsize of the axes title
    # plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

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

            # ax = sns.lineplot(x=x, y=corrs['points'][k], ax=axes[0], label=f"{model_name}, {ax_labels[k]}", color=shade, linestyle=ls)
            # axes[0].set(xlabel="Layer number", ylabel="Correlation")

            ax = sns.lineplot(x=x, y=corrs['dirs' if 'dirs' in corrs else 'points'][k], ax=axes[0], color=shade, linestyle=ls)
            axes[0].set(xlabel="Layer number", ylabel="Magnitude correlation")

            ax = sns.lineplot(x=x, y=sim_vecs[k], ax=axes[-1], label=f"{model_name}, {ax_labels[k].split(' ')[0]}", color=shade, linestyle=ls)
            axes[-1].set(xlabel="Layer number", ylabel="Direction similarity")
            axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # m, n = title.split("_")
        # m = {"proj": "projection", "steer": "steering"}[m]
        # n = {"rand": "with random directions", "sing": "with singular vectors"}[n]

    plt.suptitle(f"Causal faithfulness: {titles[title]}")
    plt.tight_layout()
    f.savefig(f"{plot_folder}/summary_{title}.png")
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

colors={"tuned": "red", "mean": "indigo", "resample": "olive", "modal": "black"}

CORR_SIZE = 20
SMALL_SIZE = 24
MEDIUM_SIZE = 24
BIGGER_SIZE = 32

plt.rc('font', size=CORR_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=CORR_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

for model_name in all_models:
    print(model_name)
    if model_name == "gpt2-large":
        continue

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
    f = plt.subplots(figsize=(6,6))
    vanilla_losses = torch.load(f"{folders['linear_oa']}/original.pth")
    lens_list = ["mean", "resample", "modal", "tuned"]
    for k in vanilla_losses:
        if k not in lens_list:
            continue
        # print(k)
        # print(list(vanilla_losses[f"{k}"]))
        sns.lineplot(list(vanilla_losses[f"{k}"]), color=colors[k], label=f"{ax_labels[k]}", linewidth=lw[i])
        # plt.show()
        plt.ylim(-0.2,5.3)
        plt.xlabel("Layer number")
        plt.ylabel("KL-divergence")
        plt.suptitle(f"Lens loss, {mn}")
        plt.tight_layout()
        plt.savefig(f"{plot_folder}/overall-{mn}.png")
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
            axes[0].set(xlabel="Layer number", ylabel="Correlation (magnitude)")

            ax = sns.lineplot(x=x, y=sim_vecs[k], ax=axes[-1], label=f"{ax_labels[k]} ({n_dirs})", linewidth=math.log(n_dirs) / 2, color=shades[j], linestyle="dashed")
            axes[-1].set(xlabel="Layer number", ylabel="Similarity (direction)")

    plt.suptitle(f"Causal faithfulness: resample basis directions, {model_name}", fontsize=24)
    plt.tight_layout()
    f.savefig(f"{overall_folder}/{model_name}_summary_resample.png")
    plt.show()
    plt.close(f)
    
