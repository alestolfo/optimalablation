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
from sys import argv
import torch.optim
import time
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from itertools import cycle
from utils.training_utils import load_model_data, LinePlot, gen_resample_perm
from torch.utils.data import DataLoader
from utils.tracing_utils import get_subject_tokens, ct_inference, ct_inference_coherence
# %%

# filter for correct prompts
if len(argv) >= 3:
    pref_node_type = argv[1]
    pref_window_size = int(argv[2])
else:
    pref_node_type = "attn"
    pref_window_size = 2

# %%
sns.set()

model_name = "gpt2-xl"
mode="fact"
ds_name = "my_facts" if mode == "fact" else "my_attributes"
ds_file = "combined" if mode == "fact" else None
ds_path = "utils/datasets/facts"
base_folder=f"results/causal_tracing/{mode}"
covs_path = f"results/lens/{model_name}/linear_oa/covs.pth"
covs = torch.load(covs_path)[0]
stds = covs.diag().sqrt() * 3

TARGET_BATCH = 20

# %%
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

# %%
train_split = 0.6
causal_layers = [i for i in range(0,48)]
ct_layers = len(causal_layers)

# %%

# AIE evals
for window_size, token_type, node_type, ablate_type in [
    (w,x,y,z) 
    for w in [
        pref_window_size
        # 0, 2, 4
    ] for x in [
        "last", 
        "last_subject",
        "all_subject"
    ] for y in [
        pref_node_type
        # "attn", 
        # "mlp"
    ] for z in [
        "oa", "gauss"
    ]
]:
    folder = f"{base_folder}/{token_type}/{node_type}/{window_size}"

    with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
        correct_prompts = pickle.load(f)
    test_start = math.ceil(train_split * len(correct_prompts))
    correct_prompts = correct_prompts[test_start:]

    data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)
    data_iter = iter(data_loader)

    # init_token = torch.randn((ct_layers + 1, d_model)).to(device)

    with open(f"{folder}/null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "rb") as f:
        null_token = pickle.load(f).to(device)

    aie_probs = []
    all_clean_probs = []
    all_corrupted_probs = []

    lp = LinePlot(["loss", "step_sz"])
    gauss = ablate_type == "gauss"
    n_tries = 1 if ablate_type == "oa" else 5

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_iter)):

            tokens, subject_pos = get_subject_tokens(batch, tokenizer, mode=mode)

            for k in range(n_tries):
                target_probs, layer_probs = ct_inference(model, tokens, subject_pos, device, causal_layers, stds if gauss else null_token, token_type, node_type, window_size, gauss)
                
                corrupted_probs = layer_probs[:,-1]
                layer_probs = layer_probs[:,1:-1]

                all_clean_probs.append(target_probs.squeeze(-1))
                all_corrupted_probs.append(torch.minimum(target_probs.squeeze(-1), corrupted_probs))
                aie_probs.append(torch.minimum(target_probs, layer_probs))

    all_clean_probs = torch.stack(all_clean_probs, dim=0)
    all_corrupted_probs = torch.stack(all_corrupted_probs, dim=0)
    aie_probs = torch.stack(aie_probs, dim=0)

    all_clean_probs = all_clean_probs.unflatten(0, (-1, n_tries)) 
    all_corrupted_probs = all_corrupted_probs.unflatten(0, (-1, n_tries)) 
    aie_probs = aie_probs.unflatten(0, (-1, n_tries)) 

    acc = all_clean_probs.mean()
    corrupted_acc = all_corrupted_probs.mean()

    torch.save(all_clean_probs, f"{folder}/{ablate_type}_clean_probs.pth")
    torch.save(all_corrupted_probs, f"{folder}/{ablate_type}_corrupted_probs.pth")
    torch.save(aie_probs, f"{folder}/{ablate_type}_aie.pth")

# %% 

# coherence evals

# for ablate_type in ["oa", "gauss"]:
null_token_folder = f"{base_folder}/all_subject/mlp/0"
n_samples_per_prompt = 5

with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
    correct_prompts = pickle.load(f)
test_start = math.ceil(train_split * len(correct_prompts))
correct_prompts = correct_prompts[test_start:]

data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)

with open(f"{null_token_folder}/null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "rb") as f:
    null_token = pickle.load(f).to(device)[[-1]]

# %%

coherence_evals = {}

for ablate_type in [None, "oa", "gauss"]:
    coherence_evals[ablate_type] = []
    data_iter = iter(data_loader)

    for i, batch in enumerate(tqdm(data_iter)):

        with torch.no_grad():
            tokens, subject_pos = get_subject_tokens(batch, tokenizer, mode=mode)
            tokens = tokens.to(device)

            repeated_tokens, completions = ct_inference_coherence(model, tokens, subject_pos, stds if ablate_type == "gauss" else null_token, ablate_type)

            repeated_tokens = tokenizer.batch_decode(repeated_tokens)
            completions = tokenizer.batch_decode(completions)

        for prompt, completion in zip(repeated_tokens, completions):
            coherence_evals[ablate_type].append({
                "prompt": prompt,
                "completion": completion
            })

for ablate_type in coherence_evals:
    coherence_evals[ablate_type] = pd.DataFrame(coherence_evals[ablate_type])
    coherence_evals[ablate_type].to_csv(f"{base_folder}/coherence-results-{ablate_type}.csv")
# %%
