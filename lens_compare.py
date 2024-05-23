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
model_name = "gpt2-xl"
dir_mode = "vanilla"

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

# %%

# Experiment 1. importance similarity

# project out of random directions
# project out of directions with highest singular values

# part 1. random directions

# projection losses pass empty object
def get_edit_losses(loss_obj, directions, save_path, perturb_type="project", n_batches=1, lens_list=['tuned', 'linear_oa', 'modal']):
    for k in lens_list:
        loss_obj[k] = {"loss": [], "sim": []}
    loss_obj['perturb'] = {"loss": [], "sim": []}

    for i, vec in enumerate(tqdm(directions)):
        assert vec.shape == (n_layers, d_model)
        causal_loss, a_sim = exp.get_causal_losses(vec, perturb_type, n_batches, lens_list)
        for lens in lens_list:
            loss_obj[lens]['loss'].append(causal_loss[lens])
            loss_obj[lens]['sim'].append(a_sim[lens])
        loss_obj["perturb"]['loss'].append(causal_loss["perturb"])

        if i % 100 == -99:
            torch.save(loss_obj, save_path)


def get_resample_losses(loss_obj, directions, n_directions=[5,10,20,50,100], n_batches=100, lens_list=['tuned', 'linear_oa', 'modal']):
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
if dir_mode in ["project", "steer", "resample"]:
    k = argv[3]
    # sing_vecs[k] has singular vecs for this lens for all layers
    # columns are the singular vectors
    # use linear_oa singular vectors for modal lens
    directions = sing_vecs[k].permute((2,0,1))

    if k == "linear_oa":
        lens_list = [k, 'modal']
    else:
        lens_list = [k]
    
    if dir_mode == "project":
        path = f"{folders[k]}/proj_losses_singular.pth"
        projection_losses = {}
        get_edit_losses(projection_losses, directions, path, n_batches=BATCHES_SINGULAR, lens_list=lens_list)
        torch.save(projection_losses, path)

        print('done with projection')

    # steering
    if dir_mode == "steer":
        steering_losses = {}
        path = f"{folders[k]}/steer_losses_singular.pth"
        get_edit_losses(steering_losses, directions * exp.retrieve_causal_mag(0.2)[:, None], path, "steer", n_batches=BATCHES_SINGULAR, lens_list=lens_list)
        torch.save(steering_losses, path)

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
        path = f"{folders['linear_oa']}/proj_losses_random.pth"
        get_edit_losses(projection_losses, directions, path, n_batches = BATCHES_RAND, lens_list=lens_list)
        torch.save(projection_losses, path)
    
    elif dir_mode == "steer_rand":
        # steering
        steering_losses = {}
        path = f"{folders['linear_oa']}/steer_losses_random.pth"
        get_edit_losses(steering_losses, directions, path, "steer", n_batches = BATCHES_RAND, lens_list=lens_list)
        torch.save(steering_losses, path)

    else:
        raise Exception("compare mode not found")