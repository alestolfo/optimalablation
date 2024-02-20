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
from training_utils import load_model_data, LinePlot

# %%

# dataset settings

means_only=True
means_by_seq_pos=False
dataset="ioi"
init_modes_path = f"pruning/modes/ioi/modes_5.pkl"
folder = "pruning_edges/modes/ioi"

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 100
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
model.cfg.use_attn_result = True

if dataset == "ioi":
    ioi_ds = datasets.load_from_disk("../plausibleablation/data/ioi/ioi")
    ioi_loader = DataLoader(ioi_ds['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
    ioi_iter = cycle(iter(ioi_loader))

# %%
# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
lr = 2e-3

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

if not means_only:
    # with open(f"pruning/modes/means_dumpster.pkl", "rb") as f:
    #     means = pickle.load(f)
    with open(init_modes_path, "rb") as f:
        init_modes = torch.stack(pickle.load(f),dim=0)

    # modal_attentions = [torch.nn.Parameter(torch.randn(n_heads, d_model).to(device)) for _ in range(n_layers)]
    modal_attentions = [torch.nn.Parameter(init_modes[i].to(device)) for i in range(n_layers)]
    modal_optimizer = torch.optim.Adam(modal_attentions, lr=lr, weight_decay=0)

    for param in model.parameters():
        param.requires_grad = False

# %%
    
def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def ablation_hook_attention_all_tokens(constants, bsz, attentions, hook):
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
    return attentions

def mean_activation_hook(means_by_seq_pos, last_token_pos, attentions, hook):
    # # ignore first token because it is crazy
    with torch.no_grad():
        if last_token_pos is not None:
            indic_sample = (torch.arange(attentions.shape[1]).repeat(attentions.shape[0],1).to(device) <= last_token_pos.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
            repr = (attentions * indic_sample)
            if means_by_seq_pos:
                early_pos = repr[:,:9].sum(dim=0) / indic_sample[:,:9].sum(dim=0)
                late_pos = (repr[:,9:].sum(dim=[0,1]) / indic_sample[:,9:].sum(dim=[0,1])).unsqueeze(0)
                activation_storage.append(torch.cat([early_pos,late_pos],dim=0))
            else:
                activation_storage.append(repr[:,1:].sum(dim=[0,1]) / indic_sample[:,1:].sum(dim=[0,1]))
        elif means_by_seq_pos:
            activation_storage.append(attentions.mean(dim=0))
        else:
            activation_storage.append(attentions.mean(dim=[0,1]))
    return attentions


# %%

lp = LinePlot(['step_size', 'total_ablation_loss'])
lp_2 = LinePlot(['magnitude'])

# %%
i = 0
j = 0

if means_only:
    if means_by_seq_pos:
        # tokens 0-9 individually
        running_means = torch.zeros((n_layers, 10, n_heads, d_model)).to(device)
    else:
        running_means = torch.zeros((n_layers, n_heads, d_model)).to(device)

model.eval()
# %%
for i in tqdm(range(100)):
    # modify depending on the dataset

    if dataset == "ioi":
        b = next(ioi_iter)
        batch = tokenizer(b['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(device)
        last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) - 1
    else:
        batch = next(owt_iter)['tokens']
    
    if means_only:
        activation_storage = []

        model_results = model.run_with_hooks(
                batch,
                fwd_hooks=[
                    *[(partial(attention_points_filter, layer_no), 
                    partial(mean_activation_hook,
                            means_by_seq_pos,
                            last_token_pos if dataset == "ioi" else None)
                        ) for layer_no in range(n_layers)],
                    ]
        )
        activation_storage = torch.stack(activation_storage, dim=0)

        with torch.no_grad():
            running_means = (i * running_means + activation_storage) / (i + 1)
    else:
        modal_optimizer.zero_grad()

        model_results = model.run_with_hooks(
                batch,
                fwd_hooks=[
                    *[(partial(attention_points_filter, layer_no), 
                    partial(ablation_hook_attention_all_tokens,
                                modal_attentions[layer_no],
                                batch_size)
                        ) for layer_no in range(n_layers)],
                    *[(partial(resid_points_filter, layer_no), 
                    partial(ablation_hook_copy_all_tokens,
                            batch_size,
                            n_heads)
                        ) for layer_no in range(n_layers)]
                    ]
        )

        if dataset == "ioi":
        # ioi
            model_results = model_results[torch.arange(model_results.shape[0]),last_token_pos.repeat(n_heads * n_layers + 1)]
            model_results = model_results[torch.arange(model_results.shape[0]), batch[torch.arange(batch.shape[0]), last_token_pos+1].repeat(n_heads * n_layers + 1)]
        
        else:
            # OWT
            model_results = model_results[:,-1]
        
        model_results = model_results.log_softmax(dim=-1)
            
        # # batch_size x vocab_size
        target_results = model_results[:batch_size].clone()

        # maybe need to fix!
        # (n_layers * n_heads) x batch_size x vocab_size
        ablated_results = model_results[batch_size:].clone().unflatten(0, (n_layers * n_heads, batch_size))

        # might need to fix this ???
        kl_losses = kl_loss(ablated_results, target_results.exp()).sum(dim=-1)

        # io_loss = target_results - ablated_results

        total_loss = kl_losses

        loss = total_loss.sum()
        # loss = model(batch).sum()
        # loss = model_results.sum()
        print(loss / n_heads / n_layers)
        loss.backward()        

        prev_modals = torch.cat(modal_attentions,dim=0).detach()

        modal_optimizer.step()

        step_sz = (torch.cat(modal_attentions, dim=0).detach()-prev_modals).norm(dim=-1).mean()

        lp.add_entry({'step_size': step_sz.item(), 'total_ablation_loss': loss.item()})
        lp_2.add_entry({'magnitude': prev_modals.norm(dim=-1).mean().item()})
        if i % 100 == 0:
            # constant loss
            sns.histplot(kl_losses.flatten().detach().cpu().numpy())
            plt.savefig(f"{folder}/kl_losses_{j}.png")
            plt.show()
            plt.close()

            if i > 0:
                # squared distance from the mean
                sns.histplot((torch.stack(modal_attentions, dim=0) - running_means).square().sum(dim=-1).flatten().log().detach().cpu().numpy())
                plt.savefig(f"{folder}/mean_dist_{j}.png")
                plt.show()
                plt.close()

                # squared norm
                sns.histplot((torch.stack(modal_attentions, dim=0)).square().sum(dim=-1).flatten().log().detach().cpu().numpy())
                plt.savefig(f"{folder}/magnitudes_{j}.png")
                plt.show()
                plt.close()

                with open(f"{folder}/modes_{j}.pkl", "wb") as f:
                    pickle.dump(modal_attentions,f)

                with open(f"{folder}/means_{j}.pkl", "wb") as f:
                    pickle.dump(running_means,f)

                lp.plot(save=f"{folder}/train_dynamics_{j}.png", mv=100)
                lp_2.plot(save=f"{folder}/magnitudes_{j}.png")
            j += 1

    i += 1
# %%
if means_only:
    with open(f"{folder}/means_dumpster.pkl", "wb") as f:
        pickle.dump(running_means,f)


# %%
