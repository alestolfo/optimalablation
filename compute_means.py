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
from utils.training_utils import load_model_data, LinePlot
from utils.task_datasets import get_task_ds
# %%
# import sys
# del sys.modules['task_datasets']
# %%
# dataset settings

means_only=True
condition_pos=True
dataset = "ioi"
counterfactual=False

if len(argv) >= 2 and argv[0].startswith("compute_means"):
    print("Loading parameters")
    dataset = argv[1]
    counterfactual = len(argv) == 3
    print(dataset, counterfactual)
else:
    print("Not loading arguments", len(argv))
folder = f"results/oca/{dataset}"

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 20
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
# model.cfg.use_attn_result = True

ablation_type = "cf" if counterfactual else "oa"
task_ds = get_task_ds(dataset, batch_size, device, ablation_type)

# %%
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model

embed_filter = lambda name: name == f"blocks.{0}.hook_resid_pre"
resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_z"
mlp_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"
final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

# %%
def sum_activation_hook(condition_pos, token_mask, activation_storage, activations, hook):
    # # ignore first token

    # so that i can pass by reference
    token_mask = token_mask[0]

    with torch.no_grad():
        sum_dim = 0 if condition_pos else [0,1]
        if isinstance(token_mask, torch.Tensor):
            # attentions have bsz x seq_pos x head_idx x d_model (4 dimensions), mlp outputs have 3 dimensions
            while len(activations.shape) > len(token_mask.shape):
                token_mask = token_mask.unsqueeze(-1)
            repr = (activations * token_mask)
            # sum of activations by sequence pos, number of samples by sequence pos
            activation_storage.append(repr.sum(dim=sum_dim))
        else:
            activation_storage.append(activations.sum(dim=sum_dim))
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

if condition_pos:
    # tokens individually at each sequence position
    running_sum = torch.zeros((1, n_layers, n_heads, int(d_model / n_heads))).to(device)
    running_mlp_sum = torch.zeros((1, n_layers+1, d_model)).to(device)
    running_samples = torch.zeros((1,), dtype=torch.int).to(device)
else:
    running_sum = torch.zeros((n_layers, n_heads, int(d_model / n_heads))).to(device)
    running_mlp_sum = torch.zeros((n_layers+1, d_model)).to(device)
    running_samples = torch.zeros((1,), dtype=torch.int).to(device)

model.eval()
# %%

def compile_means(condition_pos, previous_totals, new_activations):
    if condition_pos:
        if previous_totals.shape[0] < new_activations.shape[0]:
            new_running_sum = new_activations.clone()
            new_running_sum[:previous_totals.shape[0]] += previous_totals
            return new_running_sum
        else:
            previous_totals[:new_activations.shape[0]] += new_activations
            return previous_totals
    else:
        return previous_totals + new_activations
# %%

activation_storage = []
mlp_storage = []
token_mask_wrapper = []
fwd_hooks = [*[(partial(attention_points_filter, layer_no), 
            partial(sum_activation_hook,
                    condition_pos,
                    token_mask_wrapper,
                    activation_storage)
                ) for layer_no in range(n_layers)]]
fwd_hooks.append((embed_filter, 
        partial(sum_activation_hook,
                condition_pos,
                token_mask_wrapper,
                mlp_storage)
            ))
fwd_hooks = fwd_hooks + [*[(partial(mlp_points_filter, layer_no), 
        partial(sum_activation_hook,
                condition_pos,
                token_mask_wrapper,
                mlp_storage)
            ) for layer_no in range(n_layers)]]

for name, hook in fwd_hooks:
    model.add_hook(name, hook)

# %%

for i in tqdm(range(1000)):
    batch_data = task_ds.retrieve_batch_cf(tokenizer)
    last_token_pos = batch_data[1]
    batch = batch_data[2 if counterfactual else 0]
    
    activation_storage.clear()
    mlp_storage.clear()
    token_mask_wrapper.clear()

    with torch.no_grad():
        token_mask = torch.arange(batch.shape[1]).repeat(batch.shape[0],1).to(device)
        token_mask = (token_mask <= last_token_pos.unsqueeze(1))
        token_mask_wrapper.append(token_mask)

        model_results = model(batch)

        activation_stack = torch.stack(activation_storage, dim=1)
        mlp_stack = torch.stack(mlp_storage, dim=1)
        
        running_sum = compile_means(condition_pos, running_sum, activation_stack)
        running_mlp_sum = compile_means(condition_pos, running_mlp_sum, mlp_stack)
        running_samples = compile_means(condition_pos, running_samples, token_mask.sum(dim=0 if condition_pos else [0,1]))
    
# %%

# are we taking the mean over the counterfactual distribution?
cf_tag = "cf_" if counterfactual else ""

if means_only:
    with open(f"{folder}/means_{cf_tag}attention.pkl", "wb") as f:
        # [seq_pos, layer, head, d_head]
        pickle.dump(running_sum / running_samples.clamp(min=1)[:, None, None, None], f)
    with open(f"{folder}/means_{cf_tag}mlp.pkl", "wb") as f:
        # [seq_pos, layer, d_model]
        pickle.dump(running_mlp_sum / running_samples.clamp(min=1)[:, None, None], f)
    with open(f"{folder}/means_{cf_tag}samples.pkl", "wb") as f:
        pickle.dump(running_samples, f)
# %%
