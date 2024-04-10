# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
import datasets
from itertools import cycle
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
from sys import argv
import math
from functools import partial
import torch.optim
import time
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, LinePlot
from task_datasets import OWTConfig, IOIConfig, GTConfig
# %%
# import sys
# del sys.modules['task_datasets']
# %%
# dataset settings

folder = argv[1]

folder = "oca/gt"
# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 300
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
# model.cfg.use_attn_result = True

# %%
if folder == "oca/owt":
    task_ds = OWTConfig(owt_iter, device)
elif folder == "oca/ioi":
    task_ds = IOIConfig(batch_size, device)
elif folder == "oca/gt":
    task_ds = GTConfig(batch_size, device)
else:
    raise Exception()


# %%
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model

kl_loss = torch.nn.KLDivLoss(reduction="none")

mlp_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"
resid_pre_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
resid_mid_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_mid"
resid_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_post"
attn_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out"

# %%

def compute_mean_hook(last_token_pos, activation_storage, activations, hook):
    if not isinstance(last_token_pos, torch.Tensor):
        last_token_pos = last_token_pos * torch.zeros(activations.shape[0],).to(device)
    indic_sample = (torch.arange(activations.shape[1]).repeat(activations.shape[0],1).to(device) <= last_token_pos.unsqueeze(1))
    while len(activations.shape) > len(indic_sample.shape):
        indic_sample = indic_sample.unsqueeze(-1)
    reprs = activations * indic_sample
    early_pos = reprs[:,:9].sum(dim=0) / indic_sample[:,:9].sum(dim=0)
    late_pos = (reprs[:,9:].sum(dim=[0,1]) / indic_sample[:,9:].sum(dim=[0,1])).unsqueeze(0)
    activation_storage.append(torch.cat([early_pos,late_pos],dim=0))

def copy_hook_all_tokens(bsz, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,act[:bsz]], dim=0)
    return act

def mean_ablation_hook_layer_all_tokens(constants, bsz, activations, hook):
    constants = torch.cat([constants, constants[-1].repeat(activations.shape[1]-constants.shape[0],1)], dim=0)
    activations[-bsz:,1:] = constants[1:]
    return activations

def resample_ablation_hook_layer_all_tokens(bsz, activations, hook):
    activations[-bsz:,1:] = activations[-bsz + torch.randperm(bsz).to(device),1:]
    return activations

def mode_ablation_hook_layer_all_tokens(constants, bsz, activations, hook):
    activations[-bsz:,1:] = constants
    return activations

def zero_ablation_hook_layer_all_tokens(bsz, activations, hook):
    activations[-bsz:,1:] = 0
    return activations

def save_hook_last_token(last_token_pos, activation_storage, activations, hook):
    # if isinstance(last_token_pos, torch.Tensor):
    #     last_token_pos = last_token_pos.repeat(activations.shape[0] // batch_size - 1)
    # activation_storage.append(
    #     activations[
    #         torch.arange(batch_size, activations.shape[0]),last_token_pos
    #     ])
    activation_storage.append(activations[torch.arange(batch_size),last_token_pos])
        

def last_token_hook(last_token_mask, bsz, orig_in, hook):
    out = orig_in.unflatten(0,(-1, bsz)).permute((0,1,2,3))
    out = (out * last_token_mask.unsqueeze(-1)).sum(dim=2)
    return out

# %%

with open(f"{folder}/attn_layer_means.pkl", "rb") as f:
    attn_means = pickle.load(f)
with open(f"{folder}/mlp_layer_means.pkl", "rb") as f:
    mlp_means = pickle.load(f)
with open(f"{folder}/attn_layer_modes.pkl", "rb") as f:
    attn_modes = pickle.load(f)
with open(f"{folder}/mlp_layer_modes.pkl", "rb") as f:
    mlp_modes = pickle.load(f)
# attn_means = attn_means[:,-1]
# mlp_means = mlp_means[:,-1]

# %%
# lr=1e-2
# attn_modes = torch.nn.Parameter(attn_means[:,-1].clone())
# mlp_modes = torch.nn.Parameter(mlp_means[:,-1].clone())
# modal_optimizer = torch.optim.Adam([attn_modes, mlp_modes], lr=lr, weight_decay=0)
# for param in model.parameters():
#     param.requires_grad = False

# %%
lp = LinePlot(["loss", "attn_step_sz", "mlp_step_sz"], pref_start=0)

dist_ar = [[] for _ in range(n_layers)]

for i in tqdm(range(50)):
    # modify depending on the dataset

    # modal_optimizer.zero_grad()

    batch, last_token_pos = task_ds.next_batch(tokenizer)

    with torch.no_grad():
        last_token_mask = torch.zeros_like(batch).to(device)
        last_token_mask[torch.arange(last_token_mask.shape[0]), last_token_pos] = 1

    fwd_hooks = [
        # *[(partial(resid_pre_filter, layer_no), 
        #         partial(copy_hook_all_tokens,
        #                 batch_size)
        #             ) for layer_no in range(n_layers)],
        # *[(partial(attn_post_filter, layer_no), 
        #         # partial(zero_ablation_hook_layer_all_tokens,
        #         # partial(mode_ablation_hook_layer_all_tokens,
        #         #         attn_modes[layer_no],
        #         partial(mean_ablation_hook_layer_all_tokens,
        #                 attn_means[layer_no],
        #                 batch_size)
        #             ) for layer_no in range(n_layers)],
        # *[(partial(resid_mid_filter, layer_no), 
        #         partial(copy_hook_all_tokens,
        #                 batch_size)
        #             ) for layer_no in range(n_layers)],
        # *[(partial(mlp_post_filter, layer_no), 
        #         # partial(zero_ablation_hook_layer_all_tokens,
        #         # partial(mode_ablation_hook_layer_all_tokens,
        #         #         mlp_modes[layer_no],
        #         partial(mean_ablation_hook_layer_all_tokens,
        #                 mlp_means[layer_no],
        #                 batch_size)
        #             ) for layer_no in range(n_layers)],
        *[(partial(resid_post_filter, layer_no),
            partial(save_hook_last_token, last_token_pos, dist_ar[layer_no]))
            for layer_no in range(n_layers)],
        (f"blocks.{n_layers - 1}.hook_resid_post", 
         partial(last_token_hook, last_token_mask, batch_size))
    ]

    with torch.no_grad():
        model_results = model.run_with_hooks(
            batch,
            fwd_hooks=fwd_hooks
        ).log_softmax(dim=-1)

    continue

    orig_probs = model_results[0].clone()
    ablated_probs = model_results[1:].clone()

    # _, idx = orig_probs.topk(5, dim=-1)
    # print(tokenizer.batch_decode(idx))

    # _, idx = ablated_probs[-1].topk(5, dim=-1)
    # print(tokenizer.batch_decode(idx))

    loss = kl_loss(ablated_probs, orig_probs.exp()).sum(dim=-1)
    # print(loss.mean())        
    # sns.scatterplot(x=(torch.arange(loss.shape[0]).unsqueeze(-1) * torch.ones_like(loss).cpu()).flatten(),y= loss.flatten().detach().cpu())
    # plt.show()
    # sns.histplot(loss.flatten().detach().cpu(), bins=100)
    # plt.show()
    # continue
    total_loss = loss.sum()

    print("Loss,", loss.mean().item())
    total_loss.backward()

    old_attn_modes, old_mlp_modes = attn_modes.detach().clone(), mlp_modes.detach().clone() 
    modal_optimizer.step()

    with torch.no_grad():
        lp.add_entry({
            "loss": loss.mean().item(),
            "attn_step_sz": (attn_modes - old_attn_modes).norm(dim=-1).mean().item(),
            "mlp_step_sz": (mlp_modes - old_mlp_modes).norm(dim=-1).mean().item()
        })

    if i % -100 == -1:
        with open(f"{folder}/attn_layer_modes.pkl", "wb") as f:
            pickle.dump(attn_modes,f)
        with open(f"{folder}/mlp_layer_modes.pkl", "wb") as f:
            pickle.dump(mlp_modes,f)
        lp.plot(save=f"{folder}/mode_training.png", mv=100)

# %%

# %%
for i in range(len(dist_ar)):
    dist_ar[i] = torch.cat(dist_ar[i], dim=0)[:20000]
    print(dist_ar[i].shape)
# %%
with open(f"{folder}/baseline_resid.pkl", "wb") as f:
    pickle.dump(dist_ar, f)
# %%
# for layer_no in range(len(baseline_dist)):
#     baseline_dist[layer_no] = torch.cat(baseline_dist[layer_no], dim=0)
