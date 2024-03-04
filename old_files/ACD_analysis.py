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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, LinePlot
import json
from pathlib import Path
from greater_than.utils import get_valid_years
from greater_than.data import YearDataset


# %%

# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
model.cfg.use_attn_result = True

# ioi_ds = datasets.load_from_disk("../plausibleablation/data/ioi/ioi")
# ioi_loader = DataLoader(ioi_ds['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
# ioi_iter = cycle(iter(ioi_loader))



# # Creating our dataset
# years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
# N = batch_size  
# ds = YearDataset(years_to_sample_from, N, Path("greater_than/potential_nouns.txt"), tokenizer, balanced=False, device=device, eos=False)

# MAX_LEN = ds.good_toks.size(-1)
# END_POS = MAX_LEN - 1
# XX1_POS = ds.good_prompt.index("XX1")
# YY_POS = ds.good_prompt.index("YY")

with open("color_objects/task.json") as f:
    color_ds = json.load(f)

color_ds_cycle = cycle(color_ds['examples'][1500:])

# %%
# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
# lr = 5e-3
# lamb = 1

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

# with open("pruning/modes/ioi/modes_8.pkl", "rb") as f:
# with open("pruning/pruning_modes_ioi/modes_1.pkl", "rb") as f:
# with open("pruning/means_by_pos/ioi/means_dumpster.pkl", "rb") as f:
# with open("pruning/pruning_modes_gt/modes_2.pkl", "rb") as f:
with open("pruning/pruning_modes_color/modes_3.pkl", "rb") as f:
    # n_layers x n_heads x d_model
    modal_values = pickle.load(f)
# with open("pruning/pruning_outputs/ioi_spec_modes/train_823.pkl", "rb") as f:
# with open("pruning/pruning_modes_ioi_missing_modes/train_271.pkl", "rb") as f:
# with open("pruning/pruning_means_ioi/train_251.pkl", "rb") as f:
# with open("pruning/pruning_modes_gt/train_2.pkl", "rb") as f:
with open("pruning/pruning_modes_color/train_3.pkl", "rb") as f:
#     # n_layers x n_heads x d_model
    pruning_values = pickle.load(f)

# %%


# %%

alpha_ranking = torch.stack(pruning_values,dim=0)[:,:,0]

# %%

v, i = torch.topk(alpha_ranking.flatten(), 144, sorted=True)
indices = np.array(np.unravel_index(i.cpu().detach().numpy(), alpha_ranking.shape)).T

# %%

idx_greater_than_zero = (np.arange(v.shape[0]) * (v > 0).cpu().numpy()).argmax()
perm_heads = np.random.permutation(idx_greater_than_zero+1)
indices[:idx_greater_than_zero+1] = indices[np.random.permutation(idx_greater_than_zero+1)]

# %%

# resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"

# %%

# sample pruned heads independently from batch, or use same pruned heads for each batch item?
# currently using the former

# %%

# # n_heads x 2, first column = location (alpha), second column = scale (beta)
n_samples = 25

# %%

for param in model.parameters():
    param.requires_grad = False
model.eval()

# %%
def pruning_hook_attention_all_tokens(constants, prune_mask, bsz, attentions, hook):
    # N by 2. First column = batch item, second column = head idx
    prune_mask = prune_mask.unsqueeze(1).unsqueeze(-1)

    # for mean ablation (repeat after 10th token)
    # constants = torch.cat([constants,constants[-1].repeat(attentions.shape[1] - constants.shape[0],1,1)], dim=0)
    
    attentions[bsz:] = (1-prune_mask) * constants + prune_mask * attentions[bsz:].clone()

    # prune_idx = prune_mask.clone()
    # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
    return attentions

# %%
lp = LinePlot(['kl_loss', 'step_size', 'av_alpha', 'complexity_loss'])
torch.autograd.set_detect_anomaly(True)

i = 0
j = 0
cum_losses = []
for i in tqdm(range(200)):
    batch = next(owt_iter)['tokens'].to(device)

    # ioi
    # b = next(ioi_iter)
    # batch = tokenizer(b['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(device)
    # last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) - 1

    # greater_than
    # batch = YearDataset(years_to_sample_from, N, Path("greater_than/potential_nouns.txt"), tokenizer, balanced=False, device=device, eos=False).good_toks
    # last_token_pos = (batch.shape[1] - 1) * torch.ones(batch.shape[0], dtype=torch.int)

    # color
    batch = tokenizer(["Q: " + next(color_ds_cycle)['input'] + " A: It's a" for _ in range(batch_size)], padding=True, return_tensors='pt')['input_ids'].to(device)
    last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1)

    with torch.no_grad():
        # sampling_optimizer.zero_grad()

        # sample
        # all_sampling_params = torch.stack(sampling_params, dim=0)
        # unif = torch.rand((n_layers, batch_size * n_samples, n_heads)).to(device)
        # prune_mask = sample_mask(unif, all_sampling_params)
        prune_mask = torch.zeros(n_layers, n_samples, batch_size, n_heads).to(device)

        for k in range(n_samples):
            prune_mask[indices[:k,0],k,:,indices[:k,1]] = 1
        
        prune_mask = prune_mask.flatten(1,2)

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

    cum_losses.append(kl_losses)

# %%
cum_losses = torch.cat(cum_losses, dim=-1)
# %%
v_perm = v.clone()
# v_perm[:idx_greater_than_zero+1] = v_perm[perm_heads]
# %%
ax = sns.lineplot(x=[i for i in range(n_samples)], y=cum_losses.mean(dim=1).detach().cpu().numpy(), label="kl_loss")
ax.legend(loc="right")
sns.lineplot(x=[i for i in range(n_samples)], y=v_perm[:n_samples].detach().cpu().numpy(), ax=ax.twinx(), color="red", label="alpha")
plt.show()

# %%
sns.histplot(cum_losses[0].detach().cpu().numpy())
# %%
sns.histplot(cum_losses[0].detach().cpu().numpy())
# %%
