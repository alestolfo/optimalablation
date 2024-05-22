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
from utils.tracing_utils import get_subject_tokens, replace_subject_tokens, patch_component_last_token, patch_component_token_pos, patch_component_all_tokens

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
batch_size = 3
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
lr = 1e-3

# learning hyperparameters
kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
    correct_prompts = pickle.load(f)

# %%
# all_keys = set()
# for prompt in correct_prompts:
#     for k in prompt:
#         all_keys.add(k)

# for prompt in correct_prompts:
#     for k in prompt:
#         if prompt[k] == None:
#             prompt[k] = ''
# %%
train_split = 0.6
correct_prompts = correct_prompts[:math.ceil(train_split * len(correct_prompts))]
data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)
data_iter = cycle(iter(data_loader))

# %%

causal_layers = [i for i in range(0,48)]
ct_layers = len(causal_layers)

# init_token = torch.randn((ct_layers + 1, d_model)).to(device)

# with open(f"{folder}/null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "rb") as f:
#     init_token = pickle.load(f)
grad_acc = math.ceil(TARGET_BATCH / batch_size)
attn_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out" 
mlp_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out" 

# %%

for p in model.parameters():
    p.requires_grad = False

# %%

window_size = 0
# attn, mlp
token_pos_list = ["last_subject", "all_subject", "last"]
node_type_list = ["attn", "mlp"]

lp = LinePlot(["kl_loss", "step_sz"])

# TRAINING RUNS FOR THIS: MLP/ATTN; 0/2/4 WINDOWS; 4 token positions; KL vs train loss.
for token_type in token_pos_list:
    for node_type in node_type_list:

        init_token = torch.load(f"{folder}/subject_means.pth")
        init_token = init_token.unsqueeze(0).repeat(ct_layers+1, 1)
        null_token = torch.nn.Parameter(init_token)
        optimizer = torch.optim.AdamW([null_token], lr=lr, weight_decay=0)
        counter = 0
        n_steps = 0
        acc_loss = 0
        optimizer.zero_grad()

        for no_batches in tqdm(range(500 * grad_acc)):
            batch = next(data_iter)

            tokens, subject_pos = get_subject_tokens(batch, tokenizer, mode=mode)
            bsz = tokens.shape[0]

            if token_type == "last":
                patch_token_pos = None
            elif token_type == "last_subject":
                mask = torch.zeros_like(tokens).to(device)
                mask[subject_pos[:,0], subject_pos[:,1]] = 1
                last_subject_pos = (mask * torch.arange(mask.shape[-1]).to(device)).argmax(dim=-1)
                patch_token_pos = torch.stack([torch.arange(mask.shape[0]).to(device), last_subject_pos], dim=-1)
            elif token_type == "all_subject":
                patch_token_pos = subject_pos
            print(patch_token_pos)
            
            tokens = tokens.repeat(ct_layers + 2, 1)

            # inference: first is clean, last is corrupted
            result = model.run_with_hooks(
                tokens,
                fwd_hooks = [
                    ("hook_embed", partial(replace_subject_tokens, bsz, subject_pos, null_token)),
                    *[
                        (partial(attn_out_filter if node_type == "attn" 
                                else mlp_out_filter, layer_no), 
                        partial(patch_component_last_token, bsz, j) if token_type == "last"
                        else partial(patch_component_token_pos, bsz, j, patch_token_pos)) 
                        for j, layer_no in enumerate(causal_layers)
                    ]
                ]
            )[:,-1].log_softmax(dim=-1)

            result = result.unflatten(0, (-1, bsz))

            target_result = result[0].unsqueeze(1)
            layer_results = result[1:].permute((1,0,2))

            target_loss, target_tokens = target_result.max(dim=-1)
            target_tokens = target_tokens.flatten()
            target_mask = torch.zeros_like(target_result).to(device)
            target_mask[torch.arange(target_result.shape[0]).to(device),0,target_tokens] = 1
            
            loss = (target_loss.unsqueeze(-1) - layer_results * target_mask).relu().sum() / TARGET_BATCH
            loss.backward()

            acc_loss += loss.item()

            if counter % (-1 * grad_acc) == -1:
                n_steps += 1
                counter = 0

                prev_bias = null_token.clone()

                optimizer.step()
                optimizer.zero_grad()

                lp.add_entry({
                    "kl_loss": acc_loss / grad_acc,
                    "step_sz": (prev_bias - null_token).norm(dim=-1).mean().item(),
                })

                acc_loss = 0

                if n_steps % -10 == -1:
                    lp.plot(["kl_loss"], save=f"{folder}/{token_type}_{node_type}_train.png", mv=20, start=0)
                    lp.plot(["step_sz"], save=f"{folder}/{token_type}_{node_type}_train_step.png", mv=20, start=0)
                    with open(f"{folder}_{token_type}_{node_type}_null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "wb") as f:
                        pickle.dump(null_token, f)
            else:
                counter += 1
# %%
# lp = LinePlot(["kl_loss", "step_size"])

# no_batches = 0
# for batch in tqdm(data_iter):
#     max_length = 0
#     token_seqs = []
#     subject_token_pos = []
#     last_token_pos = []
#     for i, temp in enumerate(batch['template']):
#         pre_subject, post_subject = temp.split("{}")
#         tokens = tokenizer(pre_subject.rstrip())['input_ids']
#         subject_pos = len(tokens)
#         tokens = tokens + tokenizer(("" if pre_subject == "" else " ") + batch['subject'][i])['input_ids']
#         subject_end_pos = len(tokens)
#         tokens = tokens + tokenizer(post_subject)['input_ids']
        
#         last_token_pos.append(len(tokens))
#         max_length = max(max_length, len(tokens))
#         token_seqs.append(tokens)

#         full_prompt = temp.replace("{}", batch['subject'][i])
#         assert tokenizer(full_prompt)['input_ids'] == tokens
#         # print(tokens)
#         # print()
#         # print(tokenizer.batch_decode(tokens))
#         # print(tokenizer.batch_decode(tokenizer(full_prompt)['input_ids']))

#         # for pos in range(subject_pos, subject_end_pos):
#         #     subject_token_pos.append([i, pos])
    
#     continue
#     optimizer.zero_grad()

#     # prepend bos token
#     tokens = torch.stack([torch.nn.functional.pad(torch.tensor(seq, device=device), (1, max_length - len(seq)), "constant", tokenizer.pad_token_id) for seq in token_seqs], dim=0)

#     subject_token_pos = torch.tensor(subject_token_pos, device=device)

#     with torch.no_grad():
#         target_probs = model(tokens)[torch.arange(batch_size, device=device), last_token_pos].softmax(dim=-1)

#     no_subject_probs = model.run_with_hooks(
#         tokens,
#         fwd_hooks=[
#             ("hook_embed", partial(replace_subject_tokens, subject_token_pos, null_token))
#         ]
#     )[torch.arange(batch_size, device=device), last_token_pos].log_softmax(dim=-1)

#     loss = kl_loss(no_subject_probs, target_probs).sum(dim=-1)
#     loss = loss.mean()
#     loss.backward()

#     prev_bias = null_token.clone()

#     optimizer.step()

#     no_batches += 1
#     lp.add_entry({
#         "step_size": (null_token - prev_bias).norm().item(), 
#         "kl_loss": loss.item()
#     })

#     if no_batches % -100 == -1:
#         lp.plot(save=f"{folder}/train.png", mv=20)
#         with open(f"{folder}/null_token.pkl", "wb") as f:
#             pickle.dump(null_token, f)
#     # print(batch)

# # %%
# # pred_tokens = tokenizer.batch_decode(prediction[torch.arange(prediction.shape[0], device=device), last_token_pos])
# # # %%

# # # %%

# # with open("utils/datasets/facts/known_1000.json", "r") as f:
# #     factual_probs = json.load(f)

# # # %%

# # # %%

# # params = torch.nn.Parameter(torch.ones((50,500)).to("cuda:0"))

# # # %%
# # # AdamW([params],0.01)
# # # %%

# %%



