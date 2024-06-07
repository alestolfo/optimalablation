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
from sys import argv
from utils.tracing_utils import ct_inference, get_subject_tokens

# %%
sns.set()
if len(argv) >= 3:
    pref_node_type = argv[1]
    pref_window_size = int(argv[2])
else:
    pref_node_type = "attn"
    pref_window_size = 2
device = "cuda:0"

mode="fact"
ds_name = "my_facts" if mode == "fact" else "my_attributes"
ds_path = "utils/datasets/facts"
base_folder=f"results/causal_tracing/{mode}"

TARGET_BATCH = 20

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-xl"
batch_size = 5
clip_value = 1e5

device = torch.device(device if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)
tokenizer = model.tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-3

# %%

with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
    correct_prompts = pickle.load(f)

# %%
train_split = 0.6
correct_prompts = correct_prompts[:math.ceil(train_split * len(correct_prompts))]
data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)
data_iter = cycle(iter(data_loader))

for p in model.parameters():
    p.requires_grad = False

# %%

causal_layers = [i for i in range(0,48)]
# attn, mlp
token_pos_list = [
    # "last_subject", "all_subject",
    "last"
]
node_type_list = [pref_node_type]
window_size_list = [pref_window_size]

grad_acc = math.ceil(TARGET_BATCH / batch_size)
lp = LinePlot(["loss", "step_sz"])

# TRAINING RUNS FOR THIS: MLP/ATTN; 0/2/4 WINDOWS; 4 token positions; KL vs train loss.
for (token_type, node_type, window_size) in [
    (x,y,z) for x in token_pos_list for y in node_type_list for z in window_size_list
]:
    folder = f"{base_folder}/{token_type}/{node_type}/{window_size}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    init_token = torch.load(f"{base_folder}/subject_means.pth")

    init_token = init_token.unsqueeze(0).repeat(len(causal_layers)+1, 1).to(device)
    null_token = torch.nn.Parameter(init_token)
    optimizer = torch.optim.AdamW([null_token], lr=lr, weight_decay=0)
    counter = 0
    n_steps = 0
    acc_loss = 0
    optimizer.zero_grad()

    for no_batches in tqdm(range(1000 * grad_acc)):
        batch = next(data_iter)
        tokens, subject_pos = get_subject_tokens(batch, tokenizer, mode)

        target_probs, layer_probs = ct_inference(model, tokens, subject_pos, device, causal_layers, null_token, token_type, node_type, window_size)
        
        loss = (target_probs - layer_probs).relu().sum() / TARGET_BATCH
        loss.backward()

        acc_loss += loss.item()

        if counter % (-1 * grad_acc) == -1:
            n_steps += 1
            counter = 0

            prev_bias = null_token.clone()

            optimizer.step()
            optimizer.zero_grad()

            lp.add_entry({
                "loss": acc_loss / grad_acc,
                "step_sz": (prev_bias - null_token).norm(dim=-1).mean().item(),
            })

            acc_loss = 0

            if n_steps % -50 == -1:
                lp.plot(["loss"], save=f"{folder}/train.png", mv=20, start=0)
                lp.plot(["step_sz"], save=f"{folder}/train_step.png", mv=20, start=0)
                with open(f"{folder}/null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "wb") as f:
                    pickle.dump(null_token, f)
            if n_steps % -100 == -1:
                with open(f"{folder}/{n_steps}_null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "wb") as f:
                    pickle.dump(null_token, f)
        else:
            counter += 1