# %%
import torch
import json
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from glob import glob
from functools import partial
import os
import torch.optim
import time
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from itertools import cycle
from utils.training_utils import load_model_data, LinePlot
from torch.utils.data import DataLoader
from utils.tracing_utils import get_subject_tokens, replace_subject_tokens, gauss_subject_tokens, patch_component_last_token, patch_component_subject_tokens, patch_component_all_tokens
# %%

# filter for correct prompts

sns.set()

mode="fact"
ds_name = "my_facts" if mode == "fact" else "my_attributes"
ds_path = "utils/datasets/facts"
folder=f"results/causal_tracing/{mode}"

TARGET_BATCH = 20

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-xl"
batch_size = 10
clip_value = 1e5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)
tokenizer = model.tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

# learning hyperparameters
kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
    correct_prompts = pickle.load(f)

data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)
data_iter = iter(data_loader)
# %%

causal_layers = [i for i in range(36,44)]
ct_layers = len(causal_layers)

# init_token = torch.randn((ct_layers + 1, d_model)).to(device)

with open(f"{folder}/null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "rb") as f:
    null_token = pickle.load(f)
# init_token = torch.load(f"{folder}/subject_means.pth")
# init_token = init_token.unsqueeze(0).repeat(ct_layers+1, 1)

# %%
attn_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out" 
mlp_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out" 
# %%
covs = torch.load(f"{folder}/stds.pth")
stds = covs.diag().sqrt() * 3

# %%
aie_probs = {}
for layer in causal_layers:
    aie_probs[layer] = []
# %%
all_clean_probs = []
all_corrupted_probs = []
gauss = False
lp = LinePlot(["kl_loss", "step_sz"])

with torch.no_grad():
    for i, batch in enumerate(tqdm(data_iter)):
        tokens, subject_pos = get_subject_tokens(batch, tokenizer, mode=mode)
        bsz = tokens.shape[0]

        tokens = tokens.repeat(ct_layers + 2, 1)

        # inference: first is clean, last is corrupted
        result = model.run_with_hooks(
            tokens,
            fwd_hooks = [
                ("hook_embed", 
                    partial(gauss_subject_tokens, bsz, subject_pos, stds) if gauss else
                    partial(replace_subject_tokens, bsz, subject_pos, null_token)
                ),
                *[
                    (partial(mlp_out_filter, layer_no), 
                    partial(patch_component_last_token, bsz, j)) 
                    for j, layer_no in enumerate(causal_layers)
                ]
            ]
        )[:,-1].log_softmax(dim=-1)

        result = result.unflatten(0, (-1, bsz))

        target_result = result[0].unsqueeze(1)
        corrupted_result = result[-1].unsqueeze(1)
        layer_results = result[1:-1].permute((1,0,2))

        total_loss = kl_loss(corrupted_result, target_result.exp()).sum(dim=-1)
        aie_loss = kl_loss(layer_results, target_result.exp()).sum(dim=-1)

        # print("Total loss", total_loss.item())

        # accumulated gradients
        # loss = aie_loss.mean(dim=1).sum() / grad_acc
        # print("AIE loss", aie_loss.mean().item())

        target_tokens = target_result.argmax(dim=-1).flatten()
        bij = torch.arange(corrupted_result.shape[0]).to(device)
        probs = target_result[bij, 0, target_tokens].exp()
        corrupted_probs = corrupted_result[bij, 0, target_tokens].exp()

        all_clean_probs.append(probs)
        all_corrupted_probs.append(corrupted_probs)

        for j, layer in enumerate(causal_layers):
            layer_probs = layer_results[bij, j, target_tokens].exp()
            aie_probs[layer].append(layer_probs)

        if i > 200:
            break
# %%

all_clean_probs = torch.cat(all_clean_probs, dim=0)
all_corrupted_probs = torch.cat(all_corrupted_probs, dim=0)

# %%
sns.scatterplot(x=all_clean_probs.cpu(), y=all_corrupted_probs.cpu(), s=5)

# %%

acc = all_clean_probs.mean()
corrupted_acc = all_corrupted_probs.mean()

print(f"Clean {acc}, corrupted {corrupted_acc}")
# %%
for layer in aie_probs:
    aie_probs[layer] = torch.cat(aie_probs[layer], dim=0)

# %%
for layer in aie_probs:
    # sns.histplot(aie_probs[layer].cpu().clamp(min=0,max=1), bins=100)
    print(f"Layer {layer}, AIE", aie_probs[layer].mean().item())
# %%
