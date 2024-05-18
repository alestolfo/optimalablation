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
from sys import argv
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, save_hook_last_token, LinePlot
from utils.lens_utils import LensExperiment

# %%

sns.set()
model_name = argv[1]
folder=f"results/lens/{model_name}/oa"
shared_bias = False

# %%
# model_name = "EleutherAI/pythia-70m-deduped"

if model_name == "gpt2-xl":
    batch_size = 50
elif model_name == "gpt2-large":
    batch_size = 75
elif model_name == "gpt2-medium":
    batch_size = 100
elif model_name == "gpt2-small":
    batch_size = 200
else:
    raise Exception("Model not found")

device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 2e-3

kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

# prior_bias = [
#     model.blocks[i].attn.b_O.clone() for i in range(n_layers)
# ]
prior_bias = [
    torch.randn_like(model.blocks[i].attn.b_O) for i in range(n_layers)
]

attn_bias = [
    # torch.nn.Parameter(torch.ones((i+1, d_model)).to(device)) for i in range(n_layers)
    torch.nn.Parameter((prior_bias[i] if shared_bias else prior_bias[i].repeat(i+1,1)).to(device)) for i in range(n_layers)
]

exp = LensExperiment(model, owt_iter, {}, device, pretrained=False)
exp.all_lens_bias['modal'] = attn_bias

# %%
lp = LinePlot([*[f"kl_loss_{k}" for k in range(n_layers)], 'step_size'])

lens_optimizer = torch.optim.AdamW(attn_bias, lr=lr, weight_decay=0)

for param in model.parameters():
    param.requires_grad = False

for p in attn_bias:
    p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))

# %%
    
for i in tqdm(range(lp.t, 30000)):
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
    
    # LOGGED
    modal_lens_probs = exp.apply_modal_lens(activation_storage, shared_bias=shared_bias)

    kl_losses = kl_loss(modal_lens_probs, model_probs).sum(dim=-1).mean(dim=0)
    loss = kl_losses.sum()
    loss.backward()

    prev_weights = torch.cat(attn_bias, dim=0).detach()

    lens_optimizer.step()

    step_sz = (torch.cat(attn_bias, dim=0)-prev_weights).abs().sum()
    lp.add_entry({
        "step_size": step_sz.item(), 
        **{f"kl_loss_{k}": kl_losses[k].item() for k in range(n_layers)}
    })
    
    # lens_scheduler.step()

    if math.isnan(lp.stat_book["step_size"][-1]):
        break

    if i % -500 == -1:
        lp.plot(subplots=3, save=f"{folder}/train.png", twinx=False, mv=20)
        with open(f"{folder}/lens_bias.pkl", "wb") as f:
            pickle.dump(attn_bias, f)
# %%

