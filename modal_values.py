# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
import datasets
from itertools import cycle
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
from torch.utils.data import DataLoader
from encoders import UntiedEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, ablation_hook_copy_all_tokens, ablation_hook_attention_all_tokens, LinePlot

# %%

# model_name = "EleutherAI/pythia-70m-deduped"
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
modal_attentions = [torch.nn.Parameter(torch.randn(n_heads, d_model).to(device)) for _ in range(n_layers)]
modal_optimizer = torch.optim.SGD(modal_attentions, lr=lr, weight_decay=0)

for param in model.parameters():
    param.requires_grad = False

# %%
    
def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def ablation_hook_attention_all_tokens(constants, bsz, activation_storage, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(constants.shape[0]):
        attentions[-start:-start+n_heads,:,i] = constants[i].clone()
        start += n_heads
    
    with torch.no_grad():
        activation_storage.append(attentions[:bsz].mean(dim=[0,1]))
    return attentions


# %%

lp = LinePlot(['step_size', 'total_ablation_loss'])

# %%
i = 0
running_means = torch.zeros((n_layers, n_heads, d_model)).to(device)
while i < 1000:

    # modify depending on the dataset
    batch = next(owt_iter)['tokens']
    # batch = next(ioi_iter)['ioi_sentences']

    modal_optimizer.zero_grad()

    activation_storage = []

    model_results = model.run_with_hooks(
            batch,
            fwd_hooks=[
                *[(partial(attention_points_filter, layer_no), 
                   partial(ablation_hook_attention_all_tokens,
                            modal_attentions[layer_no],
                            batch_size,
                            activation_storage)
                    ) for layer_no in range(n_layers)],
                *[(partial(resid_points_filter, layer_no), 
                   partial(ablation_hook_copy_all_tokens,
                           batch_size,
                           n_heads)
                    ) for layer_no in range(n_layers)]
                ]
    )[:,-1].softmax(dim=-1)

    with torch.no_grad():
        running_means = (i * running_means + torch.stack(activation_storage, dim=0)) / (i + 1)

    # # batch_size x vocab_size
    target_results = model_results[:batch_size].clone()

    # (n_layers * n_heads) x batch_size x vocab_size
    ablated_results = model_results[batch_size:].clone().unflatten(0, (batch_size,n_layers * n_heads)).permute((1,0,2))

    # might need to fix this ???
    kl_losses = kl_loss(ablated_results.log(), target_results).sum(dim=-1)

    loss = kl_losses.sum()
    # loss = model(batch).sum()
    # loss = model_results.sum()
    print(loss)
    loss.backward()        

    prev_modals = torch.cat(modal_attentions,dim=0).detach()

    modal_optimizer.step()

    step_sz = (torch.cat(modal_attentions, dim=0).detach()-prev_modals).abs().sum()

    lp.add_entry({'step_size': step_sz.item(), 'total_ablation_loss': loss.item()})
    if i % 100 == 0:
        # constant loss
        sns.histplot(kl_losses.sum(dim=-1).detach().cpu().numpy())
        plt.show()

        if i > 0:

            # squared distance from the mean
            sns.histplot((torch.stack(modal_attentions, dim=0) - running_means).square().sum(dim=-1).flatten().log().detach().cpu().numpy())
            plt.show()

            # squared norm
            sns.histplot((torch.stack(modal_attentions, dim=0)).square().sum(dim=-1).flatten().log().detach().cpu().numpy())
            plt.show()

            lp.plot()

    i += 1
 
 # %%
    
with open("pruning/modes/modes_0.pkl", "wb") as f:
    pickle.dump(modal_attentions,f)

with open("pruning/modes/means.pkl", "wb") as f:
    pickle.dump(running_means,f)
# modal values on IOI dataset only
# modal values on OWT dataset
# %%
with torch.no_grad():
    batch = next(owt_iter)['tokens'].to(device)
    model_results = model.run_with_hooks(
            batch,
            fwd_hooks=[
                *[(partial(attention_points_filter, layer_no), 
                   partial(ablation_hook_attention_all_tokens,
                            running_means[layer_no],
                            batch_size,
                            activation_storage)
                    ) for layer_no in range(n_layers)],
                *[(partial(resid_points_filter, layer_no), 
                   partial(ablation_hook_copy_all_tokens,
                           batch_size,
                           n_heads)
                    ) for layer_no in range(n_layers)]
                ]
    )[:,-1].softmax(dim=-1)

    # running_means = (i * running_means + torch.stack(activation_storage, dim=0)) / (i + 1)

    # # batch_size x vocab_size
    target_results = model_results[:batch_size].clone()

    # (n_layers * n_heads) x batch_size x vocab_size
    ablated_results = model_results[batch_size:].clone().unflatten(0, (batch_size,n_layers * n_heads)).permute((1,0,2))

    # might need to fix this ???
    kl_losses = kl_loss(ablated_results.log(), target_results).sum(dim=-1)

    sns.histplot(kl_losses.sum(dim=-1).detach().cpu().numpy())
    plt.show()


# %%
