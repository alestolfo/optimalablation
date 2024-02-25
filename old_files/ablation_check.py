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
from training_utils import load_model_data, ablation_hook_attention_all_tokens, ablation_hook_copy_all_tokens, LinePlot

# %%

model_name = "gpt2-small"
batch_size = 5
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
model.cfg.use_attn_result = True

ioi_ds = datasets.load_from_disk("../plausibleablation/data/ioi/ioi")
ioi_loader = DataLoader(ioi_ds['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
ioi_iter = cycle(iter(ioi_loader))

# %%
# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
lr = 1e-3

# # learning hyperparameters
# convergence_tol = 1e-4
# similarity_tol = .05
# lr_act = 1e-4
# lr_feat = 1e-5
# updates_per_batch = 100
# relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"

# %%

with open("pruning/modes/ioi/means_8.pkl", "rb") as f:
    means = pickle.load(f)
# with open("pruning/modes/means_dumpster.pkl", "rb") as f:
#     means = pickle.load(f)

# %%
running_kl_loss = []
model.eval()

# %%

def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def ablation_hook_attention_all_tokens(constants, bsz, activation_storage, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(n_heads):
        # if attentions.shape[0] > 400:
        # print(start)
        attentions[-start:-start+bsz,:,i] = constants[i].clone()
        start -= bsz
    
    # print(attentions.shape)
    # if attentions.shape[0] > 400:
    #     sns.histplot(attentions[:bsz][attentions[:bsz].abs() > 20].detach().flatten().cpu())
    #     print((attentions[:bsz].abs() > 500).nonzero())
    #     print(attentions[:bsz][(attentions[:bsz].abs() > 500)])
        
    # ignore first token because it is crazy
    with torch.no_grad():
        activation_storage.append(attentions[:bsz,1:].mean(dim=[0,1]))
    return attentions

# %%
for i in tqdm(range(500)):
    # modify depending on the dataset
    # batch = next(owt_iter)['tokens']

    b = next(ioi_iter)
    batch = tokenizer(b['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(device)
    last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) - 1

    activation_storage = []

    with torch.no_grad():
        model_results = model.run_with_hooks(
                batch,
                fwd_hooks=[
                    *[(partial(attention_points_filter, layer_no), 
                    partial(ablation_hook_attention_all_tokens,
                                means[layer_no],
                                # torch.zeros((n_heads,)),
                                batch_size,
                                activation_storage)
                        ) for layer_no in range(n_layers)],
                    *[(partial(resid_points_filter, layer_no), 
                    partial(ablation_hook_copy_all_tokens,
                            batch_size,
                            n_heads)
                        ) for layer_no in range(n_layers)]
                    ]
        )
        
        # ioi
        model_results = model_results[torch.arange(model_results.shape[0]),last_token_pos.repeat(n_heads * n_layers + 1)]
        # model_results = model_results[torch.arange(model_results.shape[0]), batch[torch.arange(batch.shape[0]), last_token_pos+1].repeat(n_heads * n_layers + 1)]

        # OWT
        # model_results = model_results[:,-1]
        model_results = model_results.log_softmax(dim=-1)
    
        # # batch_size x vocab_size
        target_results = model_results[:batch_size].clone()

        # (n_layers * n_heads) x batch_size x vocab_size
        ablated_results = model_results[batch_size:].clone().unflatten(0, (n_layers * n_heads, batch_size))

        # might need to fix this ???
        kl_losses = kl_loss(ablated_results, target_results.exp()).sum(dim=-1)
        running_kl_loss.append(kl_losses)

# %%

agg_kl_loss = torch.cat(running_kl_loss, dim=1)

# %%
print(agg_kl_loss.mean())

# %%
sns.histplot((agg_kl_loss+.01).log().flatten().detach().cpu())

# %%

sns.histplot((agg_kl_loss > .01).nonzero()[:,0].flatten().detach().cpu(), bins=agg_kl_loss.shape[0])
# %%
sns.histplot((agg_kl_loss > .01).sum(dim=0).flatten().detach().cpu())
# %%
sns.histplot(agg_kl_loss.mean(dim=-1).flatten().detach().cpu())

# %%
(agg_kl_loss < 1).nonzero()[:100]

# %%

with open("pruning/ioi_rerun/train_11.pkl", "rb") as f:
    params = pickle.load(f)

# %%
params = torch.stack(params, dim=0)

# %%

sns.histplot(params[:,:,0].detach().flatten().cpu())

# %%

sns.histplot(params[:,:,1].detach().flatten().cpu())

# %%

sns.scatterplot(x=params[:,:,0].detach().flatten().cpu(), y=params[:,:,1].detach().flatten().cpu())
# %%
# def hook_check_spec_token(act, hook):
#     norms = act.norm(dim=-1)
#     sns.histplot(norms.detach().cpu().flatten())
#     print(norms.shape)
#     print(norms[norms > 200])
#     print((norms > 200).nonzero())
#     plt.show()
#     sns.histplot(act.abs().log().detach().cpu().flatten())
#     norms = act.abs()
#     print(norms.shape)
#     print(norms[norms > 200])
#     print((norms > 200).nonzero())
#     plt.show()
#     return act

# batch = next(owt_iter)['tokens']
# model.run_with_hooks(
#     batch,
#     fwd_hooks=[(attention_points_filter, hook_check_spec_token)]
# )
# print()

# %%

with open("pruning/modes/new_run/modes_2.pkl", "rb") as f:
    modal_attentions = pickle.load(f)

with open("pruning/modes/new_run/means_2.pkl", "rb") as f:
    means = pickle.load(f)

# %%
sns.histplot((torch.stack(modal_attentions, dim=0) - means).norm(dim=-1).flatten().detach().cpu().numpy())

# %%
sns.histplot((means).norm(dim=-1).flatten().detach().cpu().numpy())

# %%



# %%
(means.abs() > 100).nonzero()
# means[means.abs() > 10]
# %%
# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
lr = 1e-3
lamb = .1

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

with open("pruning/modes/modes_0.pkl", "rb") as f:
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
n_samples = 25

# as in the louizos paper
starting_beta = 2/3
hard_concrete_endpoints = (-0.1, 1.1)
sampling_params = [torch.nn.Parameter(
    torch.stack(
        [(20+5*torch.rand(n_heads,)).log(), torch.ones(n_heads,) * starting_beta],
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
    concrete = (((.001+unif).log() - (1-unif).log() + sampling_params[:,:,:,0])/sampling_params[:,:,:,1]).sigmoid()

    hard_concrete = ((concrete + hard_concrete_endpoints[0]) * (hard_concrete_endpoints[1] - hard_concrete_endpoints[0])).clamp(0,1)

    # n_layers x (total_samples = batch_size * n_samples) x n_heads
    return hard_concrete

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
while i < 2:
    batch = next(owt_iter)['tokens'].to(device)

    b = next(ioi_iter)
    batch = tokenizer(b['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(device)
    last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) - 1

    # if find_last_token:
    #     # full sequence includes the IO
    # else:
    #     last_token_pos = -1 * torch.ones(batch.shape[0]).to(device)


    sampling_optimizer.zero_grad()

    # sample
    all_sampling_params = torch.stack(sampling_params, dim=0)
    unif = torch.rand((n_layers, batch_size * n_samples, n_heads)).to(device)
    prune_mask = sample_mask(unif, all_sampling_params)

    

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

    # alphas already logged
    complexity_loss = (all_sampling_params[:,:,0]-all_sampling_params[:,:,1] * (math.log(-hard_concrete_endpoints[0]/hard_concrete_endpoints[1]))).sigmoid()

    loss = kl_losses.sum() + lamb * complexity_loss.sum()
    # loss = io_loss.mean() + lamb * complexity_loss.sum()

    loss.backward()

    prev_alphas = all_sampling_params[:,:,0].detach()
    prev_betas = all_sampling_params[:,:,1].detach()
    sampling_optimizer.step()

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

    if i % 200 == 10:
        sns.histplot(prune_mask.detach().flatten().cpu())
        plt.savefig(f"pruning/ioi_rerun/mask_{j}.png")
        plt.close()

        sns.histplot(kl_losses.detach().flatten().cpu())
        plt.savefig(f"pruning/ioi_rerun/io_loss_{j}.png")
        plt.close()

        if i > 0:
            lp.plot(save=f"pruning/ioi_rerun/train_{j}.png")

        with open(f"pruning/ioi_rerun/train_{j}.pkl", "wb") as f:
            pickle.dump(sampling_params, f)

        j += 1
    
    print("KL:", kl_losses.mean())
    print("Complexity:", complexity_loss.sum())

    i += 1



# %%

# 