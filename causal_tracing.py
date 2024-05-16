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
from utils.training_utils import load_model_data, LinePlot
from torch.utils.data import DataLoader

# %%

# filter for correct prompts

sns.set()

folder="results/causal_tracing"
# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-xl"
batch_size = 20
clip_value = 1e5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)
tokenizer = model.tokenizer

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-2

# learning hyperparameters
kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

with open(f"utils/datasets/facts/correct_facts_{model_name}.pkl", 'rb') as f:
    correct_prompts = pickle.load(f)

data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)
data_iter = iter(data_loader)

# %%
null_token = torch.nn.Parameter(torch.randn((d_model,)).to(device))
optimizer = torch.optim.AdamW([null_token], lr=lr, weight_decay=0)

# %%
def replace_subject_tokens(subject_token_pos, subject_token, act, hook):
    act[subject_token_pos[:,0], subject_token_pos[:,1] + 1] = subject_token

# %%
lp = LinePlot(["kl_loss", "step_size"])

no_batches = 0
for batch in tqdm(data_iter):
    max_length = 0
    token_seqs = []
    subject_token_pos = []
    last_token_pos = []
    for i, temp in enumerate(batch['template']):
        pre_subject, post_subject = temp.split("{}")
        tokens = tokenizer(pre_subject.rstrip())['input_ids']
        subject_pos = len(tokens)
        tokens = tokens + tokenizer(" " + batch['subject'][i])['input_ids']
        subject_end_pos = len(tokens)
        tokens = tokens + tokenizer(post_subject)['input_ids']
        
        last_token_pos.append(len(tokens))
        max_length = max(max_length, len(tokens))
        token_seqs.append(tokens)

        for pos in range(subject_pos, subject_end_pos):
            subject_token_pos.append([i, pos])
    
    optimizer.zero_grad()

    # prepend bos token
    tokens = torch.stack([torch.nn.functional.pad(torch.tensor(seq, device=device), (1, max_length - len(seq)), "constant", tokenizer.pad_token_id) for seq in token_seqs], dim=0)

    subject_token_pos = torch.tensor(subject_token_pos, device=device)

    with torch.no_grad():
        target_probs = model(tokens)[torch.arange(batch_size, device=device), last_token_pos].softmax(dim=-1)

    no_subject_probs = model.run_with_hooks(
        tokens,
        fwd_hooks=[
            ("hook_embed", partial(replace_subject_tokens, subject_token_pos, null_token))
        ]
    )[torch.arange(batch_size, device=device), last_token_pos].softmax(dim=-1)

    loss = kl_loss(no_subject_probs.log(), target_probs).sum(dim=-1)
    loss = loss.mean()
    loss.backward()

    prev_bias = null_token.clone()

    optimizer.step()

    no_batches += 1
    lp.add_entry({
        "step_size": (null_token - prev_bias).norm().item(), 
        "kl_loss": loss.item()
    })

    if no_batches % -100 == -1:
        lp.plot(save=f"{folder}/train.png", mv=20)
        with open(f"{folder}/null_token.pkl", "wb") as f:
            pickle.dump(null_token, f)
    # print(batch)

# %%
# pred_tokens = tokenizer.batch_decode(prediction[torch.arange(prediction.shape[0], device=device), last_token_pos])
# # %%

# # %%



# with open("utils/datasets/facts/known_1000.json", "r") as f:
#     factual_probs = json.load(f)

# # %%


# # %%

# params = torch.nn.Parameter(torch.ones((50,500)).to("cuda:0"))

# # %%
# # AdamW([params],0.01)
# # %%
