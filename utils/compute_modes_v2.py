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

means_only=True
include_mlp=True
means_by_seq_pos=True
init_modes_path = f"oca/owt/means_attention.pkl"
folder = "oca/ioi"

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 6
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
# model.cfg.use_attn_result = True

# %%
# task_ds = OWTConfig(owt_iter, device)
task_ds = IOIConfig(batch_size, device)
# task_ds = GTConfig(batch_size, device)

# %%
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
lr = 1e-2

kl_loss = torch.nn.KLDivLoss(reduction="none")

embed_filter = lambda name: name == f"blocks.{0}.hook_resid_pre"
resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_z"
mlp_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"
final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

# %%
    
def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def ablation_hook_attention_all_tokens(constants, bsz, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(n_heads):
        attentions[-start:-start+bsz,:,i] = constants[i].clone()
        start -= bsz
    return attentions

def mean_activation_hook(means_by_seq_pos, last_token_pos, activation_storage, activations, hook):
    # # ignore first token because it is crazy
    with torch.no_grad():
        if isinstance(last_token_pos, torch.Tensor):
            indic_sample = (torch.arange(activations.shape[1]).repeat(activations.shape[0],1).to(device) <= last_token_pos.unsqueeze(1))
            # activations have bsz x seq_pos x head_dim x d_model (4 dimensions), mlp outputs have 3 dimensions
            while len(activations.shape) > len(indic_sample.shape):
                indic_sample = indic_sample.unsqueeze(-1)
            repr = (activations * indic_sample)
            if means_by_seq_pos:
                early_pos = repr[:,:9].sum(dim=0) / indic_sample[:,:9].sum(dim=0)
                late_pos = (repr[:,9:].sum(dim=[0,1]) / indic_sample[:,9:].sum(dim=[0,1])).unsqueeze(0)
                activation_storage.append(torch.cat([early_pos,late_pos],dim=0))
            else:
                activation_storage.append(repr[:,1:].sum(dim=[0,1]) / indic_sample[:,1:].sum(dim=[0,1]))
        elif means_by_seq_pos:
            early_pos = activations[:,:9].mean(dim=0)
            late_pos = activations[:,9:].mean(dim=[0,1]).unsqueeze(0)
            activation_storage.append(torch.cat([early_pos,late_pos], dim=0))
        else:
            activation_storage.append(activations.mean(dim=[0,1]))
    return activations

def final_hook_all_tokens(last_token_mask, orig_in, hook):
    out = orig_in.unflatten(0, (-1, batch_size))
    out = (out * last_token_mask.unsqueeze(-1)).sum(dim=2)
    return out


# %%

lp = LinePlot(['step_size', 'total_ablation_loss'], pref_start=0)
lp_2 = LinePlot(['magnitude'])

# %%
i = 0
j = 0

if means_only:
    if means_by_seq_pos:
        # tokens 0-9 individually
        running_means = torch.zeros((n_layers, 10, n_heads, int(d_model / n_heads))).to(device)
        running_mlp_means = torch.zeros((n_layers+1, 10, d_model)).to(device)
    else:
        running_means = torch.zeros((n_layers, n_heads, int(d_model / n_heads))).to(device)
        running_mlp_means = torch.zeros((n_layers+1, d_model)).to(device)

    model.eval()
else:
    with open(f"{folder}/means_attention.pkl", "rb") as f:
        means = pickle.load(f)
    if means_by_seq_pos:
        means = means[:,-1]

    with open(init_modes_path, "rb") as f:
        init_modes = pickle.load(f)
    
    if means_by_seq_pos:
        init_modes = init_modes[:,-1]

    modal_attentions = [torch.nn.Parameter(init_modes[i].to(device)) for i in range(n_layers)]
    modal_optimizer = torch.optim.Adam(modal_attentions, lr=lr, weight_decay=0)

    for param in model.parameters():
        param.requires_grad = False

# %%
for i in tqdm(range(1000)):
    # modify depending on the dataset

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    
    if means_only:
        activation_storage = []

        fwd_hooks = [*[(partial(attention_points_filter, layer_no), 
                    partial(mean_activation_hook,
                            means_by_seq_pos,
                            last_token_pos,
                            activation_storage)
                        ) for layer_no in range(n_layers)]]
        if include_mlp:
            mlp_storage = []
            fwd_hooks.append((embed_filter, 
                    partial(mean_activation_hook,
                            means_by_seq_pos,
                            last_token_pos,
                            mlp_storage)
                        ))
            fwd_hooks = fwd_hooks + [*[(partial(mlp_points_filter, layer_no), 
                    partial(mean_activation_hook,
                            means_by_seq_pos,
                            last_token_pos,
                            mlp_storage)
                        ) for layer_no in range(n_layers)]]

        model_results = model.run_with_hooks(
                batch,
                fwd_hooks=fwd_hooks
        )
        activation_storage = torch.stack(activation_storage, dim=0)
        with torch.no_grad():
            running_means = (i * running_means + activation_storage) / (i + 1)
        if include_mlp:
            mlp_storage = torch.stack(mlp_storage, dim=0)
            with torch.no_grad():
                running_mlp_means = (i * running_mlp_means + mlp_storage) / (i + 1)

    else:
        modal_optimizer.zero_grad()
        
        with torch.no_grad():
            last_token_mask = torch.zeros_like(batch).to(device)
            last_token_mask[torch.arange(last_token_mask.shape[0]), last_token_pos] = 1

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
                        ) for layer_no in range(n_layers)],
                    (final_embed_filter,
                    partial(final_hook_all_tokens,
                                last_token_mask))
                    ]
        )
        model_results = model_results.log_softmax(dim=-1)
            
        # # batch_size x vocab_size
        target_results = model_results[0].clone()

        # maybe need to fix!
        # (n_layers * n_heads) x batch_size x vocab_size
        ablated_results = model_results[1:].clone()

        # might need to fix this ???
        kl_losses = kl_loss(ablated_results, target_results.exp()).sum(dim=-1)
        loss = kl_losses.sum()
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
            plt.savefig(f"{folder}/kl_losses.png")
            plt.show()
            plt.close()

            if i > 0:
                # squared distance from the mean
                sns.histplot((torch.stack(modal_attentions, dim=0) - means).square().sum(dim=-1).flatten().log().detach().cpu().numpy())
                plt.savefig(f"{folder}/mean_dist.png")
                plt.show()
                plt.close()

                # squared norm
                sns.histplot((torch.stack(modal_attentions, dim=0)).square().sum(dim=-1).flatten().log().detach().cpu().numpy())
                plt.savefig(f"{folder}/magnitudes.png")
                plt.show()
                plt.close()

                with open(f"{folder}/modes.pkl", "wb") as f:
                    pickle.dump(modal_attentions,f)

                lp.plot(['step_size', 'total_ablation_loss'], save=f"{folder}/train_dynamics.png")
                lp_2.plot(save=f"{folder}/magnitudes.png")
            j += 1
    i += 1
    
# %%
if means_only:
    with open(f"{folder}/means_attention.pkl", "wb") as f:
        pickle.dump(running_means,f)
    with open(f"{folder}/means_mlp.pkl", "wb") as f:
        pickle.dump(running_mlp_means,f)


# %%
