# %%
import torch
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
from itertools import cycle
from encoders import UntiedEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, pruning_hook_attention_all_tokens, LinePlot

# %%

# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
folder = "pruning/ioi_likelihood"
batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.eval()
model.cfg.use_attn_result = True

ioi_ds = datasets.load_from_disk("../plausibleablation/data/ioi/ioi")
ioi_loader = DataLoader(ioi_ds['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
ioi_iter = cycle(iter(ioi_loader))

# %%
# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
lr = 5e-5
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

# with open("pruning/modes/modes_0.pkl", "rb") as f:
#     # n_layers x n_heads x d_model
#     modal_values = pickle.load(f)
with open("pruning/modes/modes_16.pkl", "rb") as f:
#     # n_layers x n_heads x d_model
    modal_values = pickle.load(f)

# %%

# resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"

# %%

# sample pruned heads independently from batch, or use same pruned heads for each batch item?
# currently using the former

# %%

# n_heads x 2, first column = location (alpha), second column = scale (beta)
n_samples = 25

# as in the louizos paper
starting_beta = 2/3
updates_per_batch = 15
hard_concrete_endpoints = (-0.1, 1.1)
sampling_params = [torch.nn.Parameter(
    torch.stack(
        [torch.ones(n_heads,), torch.ones(n_heads,) * starting_beta],
        dim=1
    ).to(device)
) for _ in range(n_layers)]
sampling_optimizer = torch.optim.Adam(sampling_params, lr=lr, weight_decay=0)

# %%

# beta and alpha should be same shape as x, or broadcastable
# def f_concrete(x, beta, alpha):
#     return ((x.log() - (1-x).log()) * beta - alpha.log()).sigmoid()

def sample_mask(unif, sampling_params):
    sampling_params = sampling_params.unsqueeze(1)

    # back prop against log alpha
    concrete = (((.001+unif).log() - (1-unif).log() + sampling_params[:,:,:,0])/(sampling_params[:,:,:,1].relu()+.001)).sigmoid()

    hard_concrete = ((concrete + hard_concrete_endpoints[0]) * (hard_concrete_endpoints[1] - hard_concrete_endpoints[0])).clamp(0,1)

    # n_layers x (total_samples = batch_size * n_samples) x n_heads
    return hard_concrete, concrete

# %%

for param in model.parameters():
    param.requires_grad = False


# %%
# cum_prune = []
# for j in range(10):
#     all_sampling_params = torch.stack(sampling_params, dim=0)
#     unif = torch.rand((n_layers, batch_size * n_samples, n_heads))
#     prune_mask = sample_mask(unif, all_sampling_params)
#     cum_prune.append(prune_mask)

# cum_prune = torch.stack(cum_prune, dim=0).flatten()
# sns.histplot(cum_prune.detach())

# %%
lp = LinePlot(['kl_loss', 'step_size', 'av_alpha', 'complexity_loss'])
torch.autograd.set_detect_anomaly(True)

i = 0
j = 0
while i < 100000:
    # batch = next(owt_iter)['tokens'].to(device)

    b = next(ioi_iter)
    batch = tokenizer(b['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(device)
    last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) - 1

    sampling_optimizer.zero_grad()

    # sample
    all_sampling_params = torch.stack(sampling_params, dim=0)
    unif = torch.rand((n_layers, batch_size * n_samples, n_heads)).to(device)
    prune_mask, concrete = sample_mask(unif, all_sampling_params)

    with torch.no_grad():
        model_results = model.run_with_hooks(
            # first batch_size samples are targets
                batch.repeat(n_samples + 1,1),
                fwd_hooks=[
                    (partial(attention_points_filter, layer_no), 
                    partial(pruning_hook_attention_all_tokens,
                            modal_values[layer_no],
                            prune_mask[layer_no],
                            batch_size)
                    ) for layer_no in range(n_layers)
                ]
        )
        # io token
        model_results = model_results[torch.arange(model_results.shape[0]),last_token_pos.repeat(n_samples + 1)]

        # io logits
        # model_results = model_results[torch.arange(model_results.shape[0]), batch[torch.arange(batch.shape[0]), last_token_pos+1].repeat(n_samples + 1)]

        # kl div
        model_results = model_results.log_softmax(dim=-1)

        # batch_size x vocab_size
        target_results = model_results[:batch_size]

        # n_samples x batch_size x vocab_size
        ablated_results = model_results[batch_size:].unflatten(0, (n_samples,batch_size))

        kl_losses = kl_loss(ablated_results, target_results.exp()).sum(dim=-1)
        # io_loss = target_results - ablated_results

    for k in range(updates_per_batch):
    
        betas = (all_sampling_params[:,:,1].relu()+.001).unsqueeze(1)
        alphas = all_sampling_params[:,:,0].unsqueeze(1)
        concrete = concrete.clamp(.005,.995)
        likelihood = ((betas * alphas * (concrete * (1-concrete)).pow(-betas - 1)) / (alphas * concrete.pow(-betas) + (1-concrete).pow(-betas) + .01).square()).log().sum(dim=[0,2])

        expected_loss = (likelihood * kl_losses.flatten()) / (likelihood.sum()+.001)
        
        # alphas already logged
        complexity_loss = (all_sampling_params[:,:,0]-all_sampling_params[:,:,1].relu() * (math.log(-hard_concrete_endpoints[0]/hard_concrete_endpoints[1]))).sigmoid()

        loss = expected_loss.sum() + 0.1 * lamb * complexity_loss.sum()
        # loss = io_loss.mean() + lamb * complexity_loss.sum()

        loss.backward(retain_graph=True)

        prev_alphas = all_sampling_params[:,:,0].detach()
        prev_betas = all_sampling_params[:,:,1].detach()
        sampling_optimizer.step()
        sampling_optimizer.zero_grad()

    nancount = torch.stack(sampling_params, dim=0).isnan().sum()
    
    if nancount > 0:
        print("NANs", nancount)
        for param in sampling_params:
            param[param[:,1].isnan().nonzero()[:,0],1] = 2/3

    nancount = torch.stack(sampling_params, dim=0).isnan().sum()
    if nancount > 0:
        print("NANs", nancount)
        break
    
    step_sz = (torch.stack(sampling_params, dim=0)[:,:,0] - prev_alphas).abs().sum()

    lp.add_entry({"step_size": step_sz.item(), "kl_loss": kl_losses.mean().item(), "av_alpha": all_sampling_params[:,:,0].mean().item(), "complexity_loss": complexity_loss.sum().item()})

    if i % 100 == 10:
        sns.histplot(prune_mask.detach().flatten().cpu())
        plt.savefig(f"{folder}/mask_{j}.png")
        plt.close()

        sns.scatterplot(x=all_sampling_params[:,:,0].detach().flatten().cpu(), y=all_sampling_params[:,:,1].detach().flatten().cpu())
        plt.savefig(f"{folder}/params_{j}.png")
        plt.close()

        sns.histplot(kl_losses.detach().flatten().cpu())
        plt.savefig(f"{folder}/io_loss_{j}.png")
        plt.close()

        if i > 0:
            lp.plot(save=f"{folder}/train_{j}.png")

        with open(f"{folder}/train_{j}.pkl", "wb") as f:
            pickle.dump(sampling_params, f)

        j += 1
    
    print("KL:", kl_losses.mean())
    print("Complexity:", complexity_loss.sum())

    i += 1

# %%

# 