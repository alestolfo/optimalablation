# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
import pickle
from utils.training_utils import load_model_data, save_hook_last_token, LinePlot
from utils.lens_utils import LensExperiment

# %%
sns.set()
modal_lens_folder="results/modal_lens/random_init"
lm_lens_folder="results/modal_lens/linear_oca"
tuned_lens_folder = "results/tuned_lens"
folder="results/modal_lens/grad_baseline"
shared_bias = False

# %%
model_name = "gpt2-small"
batch_size = 40
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
final_embed_filter = lambda name: name == f"blocks.{n_layers - 1}.hook_resid_post"

# %%

# causal_params = [torch.nn.Parameter(torch.zeros((d_model,)).to(device)) for _ in range(n_layers)]
# dummy_optimizer = torch.optim.AdamW(causal_params, weight_decay=0)

causal_params = torch.zeros((n_layers, d_model)).to(device)

# %%
# std: scalar or shape [d_model,]
def causal_and_save_hook_last_token(bsz, param, act, hook):
    # norm = torch.randn_like(act[-bsz:,-1,:]).to(device) * std

    act = torch.cat([act, act[:bsz]], dim=0)
    act[-bsz:,-1,:] = act[-bsz:,-1,:] + param
    # save_to.append(act[-bsz:,-1,:])
    return act

def final_hook_last_token(activation_storage, act, hook):
    output_ts = act[:,-1].sum(dim=0) / bsz
    activation_storage.append(output_ts)

# %%

def compute_jacobians(causal_params):
    activation_storage = []
    model.run_with_hooks(
            batch,
            fwd_hooks=[
                    *[(partial(resid_points_filter, layer_no), 
                    partial(causal_and_save_hook_last_token, bsz, causal_params[layer_no])) 
                    for layer_no in range(n_layers)],
                    (final_embed_filter, 
                     partial(final_hook_last_token, activation_storage))
                ]
    )[:,-1].softmax(dim=-1)
    return activation_storage[0]

# %%

cum_grad = torch.zeros((d_model, n_layers, d_model)).to(device)
for i in tqdm(range(1000)):
    grad_storage = []

    batch = next(owt_iter)['tokens']
    bsz = batch.shape[0]

    with torch.no_grad():
        grad = jacobian(compute_jacobians, causal_params)

    cum_grad = (i * cum_grad + grad) / (i+1)

    if i % -100 == -1:
        torch.save(cum_grad, f"{folder}/cum_grad.pth")

# %%

cum_grad = torch.load(f"{folder}/cum_grad.pth")

# %%

lr = 1e-2

lens_weights = cum_grad.permute((1,0,2))
lens_bias = [torch.nn.Parameter(torch.randn(d_model,).to(device)) for _ in range(n_layers)]
lens_optimizer = torch.optim.AdamW([*lens_weights, *lens_bias], lr=lr, weight_decay=1e-3)

for param in model.parameters():
    param.requires_grad = False

for p in lens_bias:
    p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))

exp = LensExperiment(model, owt_iter, {}, device, pretrained=False)
exp.all_lens_weights['grad'] = lens_weights
exp.all_lens_bias['grad'] = lens_bias

# %%

tuned_loss_series = [f"kl_loss_{k}" for k in range(n_layers)]
lp = LinePlot([*tuned_loss_series, 'step_size'])
    
i = 0
for i in tqdm(range(50000)):
    batch = next(owt_iter)['tokens']
    lens_optimizer.zero_grad()

    activation_storage = []

    model_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[
                *[(partial(resid_points_filter, layer_no), 
                   partial(save_hook_last_token, activation_storage),
                    ) for layer_no in range(n_layers)],
                ]
    )[:,-1].softmax(dim=-1).unsqueeze(1)

    lens_probs = exp.apply_lens('grad', activation_storage)

    kl_losses = kl_loss(lens_probs.log(), model_probs).sum(dim=-1).mean(dim=0)
    
    loss = kl_losses.sum()
    loss.backward()

    prev_bias = torch.stack(lens_bias, dim=0).detach()

    lens_optimizer.step()

    step_sz = (torch.stack(lens_bias, dim=0)-prev_bias).abs().sum()
    lp.add_entry({
        "step_size": step_sz.item(), 
        **{f"kl_loss_{k}": kl_losses[k].item() for k in range(n_layers)}
    })

    if math.isnan(lp.stat_book["step_size"][-1]):
        break

    if i % -500 == -1:
        lp.plot(series=tuned_loss_series, subplots=3, save=f"{folder}/train.png", twinx=False, mv=20)
        with open(f"{folder}/lens_weights.pkl", "wb") as f:
            pickle.dump(lens_weights, f)
        with open(f"{folder}/lens_bias.pkl", "wb") as f:
            pickle.dump(lens_bias, f)
    
    i += 1


# %%
