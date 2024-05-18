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
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv
import pickle
from utils.lens_utils import LensExperiment
from utils.training_utils import load_model_data, save_hook_last_token, LinePlot

# %%
sns.set()
# model_name = "EleutherAI/pythia-70m-deduped"
# model_name = "gpt2-medium"
model_name = argv[1]
folder=f"results/lens/{model_name}/tuned"

if model_name == "gpt2-xl":
    batch_size = 40
elif model_name == "gpt2-large":
    batch_size = 60
elif model_name == "gpt2-medium":
    batch_size = 80
elif model_name == "gpt2-small":
    batch_size = 150
else:
    raise Exception("Model not found")

device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-2

kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

exp = LensExperiment(model, owt_iter, {}, device, pretrained=False)

# %%

lens_weights = [torch.nn.Parameter(torch.randn(d_model, d_model).to(device)) for _ in range(n_layers)]
lens_bias = [torch.nn.Parameter(torch.randn(d_model,).to(device)) for _ in range(n_layers)]
lens_optimizer = torch.optim.AdamW([*lens_weights, *lens_bias], lr=lr, weight_decay=0)

for param in model.parameters():
    param.requires_grad = False

for p in lens_weights:
    p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))

for p in lens_bias:
    p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))

exp.all_lens_weights['tuned'] = lens_weights
exp.all_lens_bias['tuned'] = lens_bias

# %%

tuned_loss_series = [f"kl_loss_{k}" for k in range(n_layers)]
lp = LinePlot([*tuned_loss_series, 'step_size'])
    
i = 0
for i in tqdm(range(30000)):
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

    lens_probs = exp.apply_lens("tuned", activation_storage)

    kl_losses = kl_loss(lens_probs, model_probs).sum(dim=-1).mean(dim=0)
    loss = kl_losses.sum()
    loss.backward()

    prev_weights = torch.stack(lens_weights, dim=0).detach()

    lens_optimizer.step()

    step_sz = (torch.stack(lens_weights, dim=0)-prev_weights).abs().sum()
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

# # Confidence probes
    
# # Question 1. what makes models confident in the beginning?
# # Avg contribution from later attention head -- is it smaller in magnitude, systematically?

# # Question 2. is there suppression and does it occur in later layer or before?
# # If before, we can try to predict it with a probe.

# probe_lr = 1e-3
    
# probe_direction = torch.nn.Parameter(torch.randn(n_layers, d_model).to(device))
# probe_bias = torch.nn.Parameter(torch.randn(n_layers,).to(device))

# probe_optimizer = torch.optim.SGD([probe_direction, probe_bias], lr=probe_lr, weight_decay=0)

# # %%
# torch.autograd.set_detect_anomaly(True)

# # %%
# lp = LinePlot([f"probe_loss_{i}" for i in range(n_layers)])
# i = 0
# while i < 1000:
#     batch = next(owt_iter)['tokens']

#     with torch.no_grad():
#         tuned_lens_acc, activation_storage = get_tuned_lens_loss(batch)

#     tuned_lens_acc = tuned_lens_acc.detach()
#     tuned_lens_acc.requires_grad = True

#     activation_storage = torch.stack(activation_storage, dim=0).detach()
#     activation_storage.requires_grad = True
#     # n_layers x batch_size
#     err_estimate = einsum("n_layer d_model, n_layer batch_size d_model -> n_layer batch_size", probe_direction, activation_storage) + probe_bias.unsqueeze(1)

#     # loss = (probe_direction - 1).square()
#     loss = (err_estimate - tuned_lens_acc).abs()
#     loss.sum().backward()

#     lp.add_entry({f"probe_loss_{i}": loss[i].mean().item() for i in range(n_layers)})

#     probe_optimizer.step()

#     print(probe_direction.isnan().sum())

#     if i % 100 == 0:
#         lp.plot(twinx=False)
#     i += 1

# # activation_storage = []

# # # get the result at unembed
# # target_probs = model.run_with_hooks(
# #             batch,
# #             fwd_hooks=[
# #                 (partial(resid_points_filter, layer_no), 
# #                    partial(tuned_lens_hook,
# #                            activation_storage,
# #                            tuned_lens_weights[layer_no],
# #                            tuned_lens_bias[layer_no])
# #                     ) for layer_no in range(n_layers)
# #                 ]
# #     )[:,-1].softmax(dim=-1)

# # fit beta model?




# # %%

# %%
