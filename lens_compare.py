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

folders = {
    "modal": "results/lens/oa",
    "linear_oa": "results/lens/linear_oa",
    "tuned": "results/lens/tuned",
    "grad": "results/lens/grad"
}
shared_bias = False
# n_layers = 12

BATCHES_PER_DIR = 8
CAUSAL_BATCH_SIZE = 50
BATCHES_RESAMPLE = 40
N_RAND_DIRS = 2000

dir_mode = argv[1]

# when i don't want to load the model
n_layers = 12
# %%
model_name = "gpt2-small"
batch_size = CAUSAL_BATCH_SIZE
# 100K OWT samples with default sequence length: 235134
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

# %%

exp = LensExperiment(model, owt_iter, folders, device)
# %%
lens_list = ["modal", "linear_oa", "tuned", "grad"]

# get vanilla losses
if not os.path.exists(f"{folders['linear_oa']}/original.png"):
    exp.get_vanilla_losses(lens_list, pics_folder=folders['linear_oa'])

# %%

# get causal perturb losses
if not os.path.exists(f"{folders['linear_oa']}/causal_losses.pth"):
    exp.get_causal_perturb_losses(lens_list, save=f"{folders['linear_oa']}/causal_losses.pth", pics_folder=folders['linear_oa'])

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
        assert diff_eye.diag().abs().mean() < 5e-4
        assert diff_eye.abs().mean() < 1e-4
    # print(sing_vecs[k].shape)

# %%

# Experiment 1. importance similarity

# project out of random directions
# project out of directions with highest singular values

# part 1. random directions

# projection losses pass empty object
def get_edit_losses(loss_obj, directions, perturb_type="project", n_batches=1, lens_list=['tuned', 'grad', 'linear_oa', 'modal']):
    for k in lens_list:
        loss_obj[k] = {"loss": [], "sim": []}
    loss_obj['perturb'] = {"loss": [], "sim": []}

    for vec in tqdm(directions):
        assert vec.shape == (n_layers, d_model)
        causal_loss, a_sim = exp.get_causal_losses(vec, perturb_type, n_batches, lens_list)
        for lens in lens_list:
            loss_obj[lens]['loss'].append(causal_loss[lens])
            loss_obj[lens]['sim'].append(a_sim[lens])
        loss_obj["perturb"]['loss'].append(causal_loss["perturb"])

def get_resample_losses(loss_obj, directions, n_directions=[5,10,20,50,100], n_batches=100, lens_list=['tuned', 'grad', 'linear_oa', 'modal']):
    for k in lens_list:
        loss_obj[k] = {"loss": [], "sim": []}
    loss_obj['perturb'] = {"loss": [], "sim": []}

    for n in n_directions:
        vecs = directions[:n].permute((1,0,2))
        assert vecs.shape == (n_layers, n, d_model)

        causal_loss, a_sim = exp.get_causal_losses(vecs, "resample", n_batches, lens_list)
        for lens in lens_list:
            loss_obj[lens]['loss'].append(causal_loss[lens])
            loss_obj[lens]['sim'].append(a_sim[lens])
        loss_obj["perturb"]['loss'].append(causal_loss["perturb"])

# singular vectors projection
# for k in ['linear_oa', 'tuned', 'grad']:
k = "linear_oa"
if dir_mode in ["project", "steer", "resample"]:
    # sing_vecs[k] has singular vecs for this lens for all layers
    # columns are the singular vectors
    # use linear_oa singular vectors for modal lens
    directions = sing_vecs[k].permute((2,0,1))

    if k == "linear_oa":
        lens_list = [k, 'modal']
    else:
        lens_list = [k]
    
    if dir_mode == "project":
        projection_losses = {}
        get_edit_losses(projection_losses, directions, n_batches=BATCHES_PER_DIR, lens_list=lens_list)
        torch.save(projection_losses, f"{folders[k]}/proj_losses_singular.pth")

        print('done with projection')

    # steering
    if dir_mode == "steer":
        steering_losses = {}
        get_edit_losses(steering_losses, directions * exp.retrieve_causal_mag(0.2)[:, None], "steer", n_batches=BATCHES_PER_DIR, lens_list=lens_list)
        torch.save(steering_losses, f"{folders[k]}/steer_losses_singular.pth")

        print('done with steering')

    if dir_mode == "resample":
        resample_losses = {}
        get_resample_losses(resample_losses, directions, n_batches=BATCHES_RESAMPLE, lens_list=lens_list)
        torch.save(resample_losses, f"{folders[k]}/resample_losses_singular.pth")

        print('done with resample')
# %%
else:

    # random directions
    directions = einsum("n_layers d_mvn d_norm, batch n_layers d_norm -> batch n_layers d_mvn", 
                        exp.a_mtrx, 
                        torch.randn((N_RAND_DIRS, n_layers, d_model)).to(device))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    directions = directions * exp.retrieve_causal_mag(0.2)[:, None]

    if dir_mode == "proj_rand":
        # projection
        projection_losses = {}
        get_edit_losses(projection_losses, directions, n_batches = 2)
        torch.save(projection_losses, f"{folders['linear_oa']}/proj_losses_random.pth")
    
    else:
        # steering
        steering_losses = {}
        get_edit_losses(steering_losses, directions, "steer", n_batches = 2)
        torch.save(steering_losses, f"{folders['linear_oa']}/steer_losses_random.pth")

exit()

# %%

def summarize_loss_obj(proj_losses_file, title, separate_vecs=False):
    lens_list = ['modal', 'tuned', 'grad', 'linear_oa']

    if separate_vecs:
        proj_losses = {}
        for typ in ['tuned', 'grad', 'linear_oa']:
            proj_losses[typ] = torch.load(f"{folders[typ]}/{proj_losses_file}.pth")
            for k in proj_losses[typ]:
                # modal loss is included in linear_oa
                proj_losses[typ][k]['loss'] = torch.stack(proj_losses[typ][k]['loss'], dim=0)
                if k != "perturb":
                    proj_losses[typ][k]['sim'] = torch.stack(proj_losses[typ][k]['sim'], dim=0)
    else:
        proj_losses = torch.load(f"{folders['linear_oa']}/{proj_losses_file}.pth")
        for k in proj_losses:
            proj_losses[k]['loss'] = torch.stack(proj_losses[k]['loss'], dim=0)
        for k in lens_list:
            proj_losses[k]['sim'] = torch.stack(proj_losses[k]['sim'], dim=0)

    num_lens = len(lens_list)
    plot_list = ["points", "dirs", "a_sim"]
    figs = {}
    axes = {}
    corrs = {}
    sim_vecs = {}
    offset = 1

    for p in plot_list:
        figs[p], axes[p] = plt.subplots(n_layers-offset, num_lens, figsize=(num_lens * 5, (n_layers-offset) * 5))
        corrs[p] = {}

    for i,k in enumerate(lens_list):
        corrs['points'][k] = []
        corrs['dirs'][k] = []
        sim_vecs[k] = []

        print(k)

        if separate_vecs:
            # info for modal lens is stored inside linear_oa
            dict_key = 'linear_oa' if k == 'modal' else k
            perturb_loss = proj_losses[dict_key]['perturb']['loss']
            lens_result = proj_losses[dict_key][k]
        else:
            perturb_loss = proj_losses['perturb']['loss']
            lens_result = proj_losses[k]

        for j in range(offset,n_layers):
            model_loss = perturb_loss[...,j]

            corr = plot_no_outliers(sns.histplot, 0, 
                            model_loss.log().flatten().cpu(),
                            lens_result['loss'][...,j].log().flatten().cpu(), 
                            axes['points'][j-offset,i], xy_line=True,
                            args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
            corrs['points'][k].append(corr)

            corr = plot_no_outliers(sns.histplot, 0, 
                            model_loss.mean(dim=1).log().flatten().cpu(),
                            lens_result['loss'][...,j].mean(dim=1).log().flatten().cpu(), 
                            axes['dirs'][j-offset,i], xy_line=True,
                            args={"x": "model loss", "y": f"{k} lens loss", "corr": True})
            corrs['dirs'][k].append(corr)

            # large_ls_idx = ((model_loss.flatten() > model_loss.quantile(.5)) * (lens_result['loss'][...,j].flatten() > lens_result['loss'][...,j].quantile(.5))).nonzero()[:,0]
            # sim_vec = lens_result['sim'][...,j].flatten()[large_ls_idx]
            # sim_vecs[k].append(sim_vec.mean().item())

            plot_no_outliers(sns.histplot, 0, 
                            model_loss.log().flatten().cpu(),
                            # model_loss.log().flatten()[large_ls_idx].cpu(),
                            lens_result['sim'][...,j].flatten().cpu(), 
                            # sim_vec.cpu(), 
                            axes['a_sim'][j-offset,i],
                            args={"x": "model loss", "y": f"{k} similarity", "corr": True})

    for p in plot_list:
        figs[p].savefig(f"{folders['linear_oa']}/{title}_importance_{p}.png")
        figs[p].show()
    
    f, axes = plt.subplots(1,3, figsize=(15,5))

    for i,k in enumerate(lens_list):
        if separate_vecs:
            dict_key = 'linear_oa' if k == 'modal' else k
            lens_result = proj_losses[dict_key][k]
        else:
            lens_result = proj_losses[k]

        x = np.arange(n_layers-offset) + offset

        sns.lineplot(x=x, y=corrs['points'][k], ax=axes[0], label=k)
        sns.lineplot(x=x, y=corrs['dirs'][k], ax=axes[1], label=k)
        # ax=sns.histplot(proj_losses[k]['sim'].mean(dim=1).flatten().cpu(), ax=axes[i])
        # ax.set(**{"xlabel": f"similarity {k}", "ylabel": f"density"})

        # sns.lineplot(lens_result['sim'].mean(dim=[0,1]).flatten().cpu(), ax=axes[-1], label=k)
        sns.lineplot(x=x, y=sim_vecs[k], ax=axes[-1], label=k)
    f.savefig(f"{folders['linear_oa']}/{title}_sim_comp.png")
    f.show()

# %%
summarize_loss_obj("proj_losses_random", "proj_rand")

# %%
summarize_loss_obj("steer_losses_random", "steer_rand")

# %%
summarize_loss_obj("proj_losses_singular", "proj_sing", True)

# %%
summarize_loss_obj("steer_losses_singular", "steer_sing", True)

# %%



# %%
# Experiment 2. stimulus-response

# Degradation of performance

# CBE with resample ablation

# %%
