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
import pickle
from utils.training_utils import load_model_data, save_hook_last_token, LinePlot
from utils.lens_utils import apply_lmlp_lens

# %%
sns.set()
folder="results/modal_lens/lmlp"
shared_bias = False
# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 200
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-3

kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

if shared_bias:
    prior_bias = torch.randn_like(model.blocks[n_layers - 1].attn.b_O)
else:
    prior_bias = torch.randn((n_layers, *model.blocks[n_layers - 1].attn.b_O.shape))

attn_bias = torch.nn.Parameter(prior_bias.to(device))

lens_optimizer = torch.optim.AdamW([attn_bias], lr=lr, weight_decay=0)

for param in model.parameters():
    param.requires_grad = False

for p in attn_bias:
    p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))

# %%
# modal lens train
lp = LinePlot([*[f"kl_loss_{k}" for k in range(n_layers)], 'step_size'])
    
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

    modal_lens_probs = apply_lmlp_lens(model, attn_bias, activation_storage, n_layers, shared_bias)
    
    kl_losses = kl_loss(modal_lens_probs.log(), model_probs).sum(dim=-1).mean(dim=0)
    
    loss = kl_losses.sum()
    loss.backward()

    prev_weights = attn_bias.detach()

    lens_optimizer.step()

    step_sz = (attn_bias-prev_weights).abs().sum()
    lp.add_entry({
        "step_size": step_sz.item(), 
        **{f"kl_loss_{k}": kl_losses[k].item() for k in range(n_layers)}
    })

    # lens_scheduler.step()

    if math.isnan(lp.stat_book["step_size"][-1]):
        break

    if i % -500 == -1:
        lp.plot(subplots=3, save=f"{folder}/train.png", twinx=False, mv=20)
        with open(f"{folder}/lens_weights.pkl", "wb") as f:
            pickle.dump(attn_bias, f)
# %%
