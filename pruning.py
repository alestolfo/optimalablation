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
from encoders import UntiedEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, pruning_hook_attention_all_tokens, LinePlot

# %%

# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 20
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
lr = 1e-4
lamb = 1

# # learning hyperparameters
# convergence_tol = 1e-4
# similarity_tol = .05
# lr_act = 1e-4
# lr_feat = 1e-5
# updates_per_batch = 100
# relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

# %%

# import modal values

with open("pruning/modal_values.pkl") as f:
    # n_layers x n_heads x d_model
    modal_values = pickle.load(f)

# %%

# resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"

# %%

# sample pruned heads independently from batch, or use same pruned heads for each batch item?
# currently using the former

# %%

# n_heads x 2, first column = location (alpha), second column = scale (beta)
n_samples = 20

# as in the louizos paper
starting_beta = 2/3
hard_concrete_endpoints = (-0.1, 1.1)
sampling_params = [torch.nn.Parameter(
    torch.stack(
        [torch.rand(n_heads,), torch.ones(n_heads,) * starting_beta],
        dim=1
    )
) for _ in range(n_layers)]
sampling_optimizer = torch.optim.SGD(sampling_params, lr=lr, weight_decay=0)

# %%

# beta and alpha should be same shape as x, or broadcastable
# def f_concrete(x, beta, alpha):
#     return ((x.log() - (1-x).log()) * beta - alpha.log()).sigmoid()

def sample_mask(unif, sampling_params):
    concrete = ((unif.log() - (1-unif).log() + sampling_params[:,0].log())/sampling_params[:,1]).sigmoid()
    hard_concrete = ((concrete - hard_concrete_endpoints[0]) * (hard_concrete_endpoints[1] - hard_concrete_endpoints[0])).clamp(0,1)

    # n_layers x (total_samples = batch_size * n_samples) x n_heads
    return hard_concrete

# %%

for param in model.parameters():
    param.requires_grad = False

# %%

i = 0
while i < 1000:
    batch = next(owt_iter)['tokens']
    sampling_optimizer.zero_grad()

    # sample
    all_sampling_params = torch.stack(sampling_params, dim=0)
    unif = torch.rand((n_layers, batch_size * n_samples, n_heads))
    prune_mask = sample_mask(unif, all_sampling_params)

    model_results = model.run_with_hooks(
        # first batch_size samples are targets
            batch.repeat(n_samples + 1),
            fwd_hooks=[
                (partial(attention_points_filter, layer_no), 
                   partial(pruning_hook_attention_all_tokens,
                           modal_values[layer_no],
                           prune_mask[layer_no])
                ) for layer_no in range(n_layers)
            ]
    )[:,-1].softmax(dim=-1)

    # batch_size x vocab_size
    target_results = model_results[:batch_size]

    # n_samples x batch_size x vocab_size
    ablated_results = model_results[batch_size:].unflatten(0, (n_samples,batch_size))

    kl_losses = kl_loss(ablated_results.log(), target_results).sum(dim=-1)

    complexity_loss = (all_sampling_params[:,0].log()-all_sampling_params[:,1] * (math.log(-hard_concrete_endpoints[0]/hard_concrete_endpoints[1]))).sum()

    loss = kl_losses + lamb * complexity_loss
    loss.backward()
    sampling_optimizer.step()

    


