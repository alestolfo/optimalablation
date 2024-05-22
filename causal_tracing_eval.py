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
from utils.training_utils import load_model_data, LinePlot, gen_resample_perm
from torch.utils.data import DataLoader
from utils.tracing_utils import get_subject_tokens, replace_subject_tokens, gauss_subject_tokens, patch_component_last_token, patch_component_subject_tokens, patch_component_all_tokens
# %%

# filter for correct prompts

sns.set()

mode="fact"
ds_name = "my_facts" if mode == "fact" else "my_attributes"
ds_path = "utils/datasets/facts"
base_folder=f"results/causal_tracing/{mode}"

TARGET_BATCH = 20

# %%
# with open("utils/datasets/facts/my_attributes.pkl", "rb") as f:
#     attributes_ds = pickle.load(f)

# correct_prompts = []
# subject_object_pairs = set()
# rel_names = {}
# for a_line in attributes_ds:
#     a_line["attribute"] = a_line["object"]
#     del a_line["object"]
#     if (a_line["subject"], a_line["attribute"]) in subject_object_pairs:
#         # print("dupe")
#         continue
#     if a_line['relation_name'] not in rel_names:
#         rel_names[a_line['relation_name']] = 1
#     else:
#         rel_names[a_line['relation_name']] += 1
#     if rel_names[a_line['relation_name']] >= 40:
#         continue
#     subject_object_pairs.add((a_line["subject"], a_line["attribute"]))
#     correct_prompts.append(a_line)

# with open("utils/datasets/facts/my_facts_old.pkl", "rb") as f:
#     facts_ds = pickle.load(f)

# for a_line in facts_ds:
#     if (a_line["subject"], a_line["attribute"]) in subject_object_pairs:
#         print("dupe")
#         print(a_line["subject"], a_line["attribute"])
#         continue
#     subject_object_pairs.add((a_line["subject"], a_line["attribute"]))
#     correct_prompts.append(a_line)

# # %%
# with open("utils/datasets/facts/my_facts.pkl", "wb") as f:
#     pickle.dump(correct_prompts, f)

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

init_token = torch.load(f"{base_folder}/subject_means.pth")

# %%
with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
    correct_prompts = pickle.load(f)

data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)
# %%

causal_layers = [i for i in range(0,48)]
ct_layers = len(causal_layers)

# %%
attn_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out" 
mlp_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out" 
# %%
covs = torch.load(f"{base_folder}/stds.pth")
stds = covs.diag().sqrt() * 3

# %% 
gauss = False
node_type = "attn"
folder = f"{base_folder}/{node_type}"
ablate_type = "_gauss" if gauss else ""
data_iter = iter(data_loader)
aie_probs = {}
for layer in causal_layers:
    aie_probs[layer] = []
# %%

with open(f"{folder}/null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "rb") as f:
    null_token = pickle.load(f)
# init_token = torch.load(f"{folder}/subject_means.pth")
# init_token = init_token.unsqueeze(0).repeat(ct_layers+1, 1)

null_token = null_token[20].repeat((null_token.shape[0],1))

# %%
all_clean_probs = []
all_corrupted_probs = []
all_baseline_diffs = []
lp = LinePlot(["kl_loss", "step_sz"])

with torch.no_grad():
    for i, batch in enumerate(tqdm(data_iter)):
        tokens, subject_pos = get_subject_tokens(batch, tokenizer, mode=mode)
        bsz = tokens.shape[0]

        permutation = gen_resample_perm(null_token.shape[0])

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
                    (partial(attn_out_filter if node_type == "attn" else mlp_out_filter, layer_no), 
                    partial(patch_component_last_token, bsz, j)) 
                    for j, layer_no in enumerate(causal_layers)
                ]
            ]
        )[:,-1].log_softmax(dim=-1)

        result = result.unflatten(0, (-1, bsz))

        target_result = result[0].unsqueeze(1)
        layer_results = result[1:-1].permute((1,0,2))

        target_tokens = target_result.argmax(dim=-1).flatten()

        bij = torch.arange(target_result.shape[0]).to(device)
        probs = target_result[bij, 0, target_tokens].exp()
        all_clean_probs.append(probs)

        for j, layer in enumerate(causal_layers):
            layer_probs = layer_results[bij, j, target_tokens].exp()
            aie_probs[layer].append(layer_probs)

        corrupted_result = result[-1].unsqueeze(1)
        corrupted_probs = corrupted_result[bij, 0, target_tokens].exp()
        # if gauss:
        # else:
        #     corrupted_result = model.run_with_hooks(
        #         tokens,
        #         fwd_hooks = [
        #             ("hook_embed", 
        #                 partial(replace_subject_tokens, bsz, subject_pos, null_token)
        #             )
        #         ]
        #     )[:,-1].log_softmax(dim=-1)


        # total_loss = kl_loss(corrupted_result, target_result.exp()).sum(dim=-1)
        # aie_loss = kl_loss(layer_results, target_result.exp()).sum(dim=-1)

        # print("Total loss", total_loss.item())

        # accumulated gradients
        # loss = aie_loss.mean(dim=1).sum() / grad_acc
        # print("AIE loss", aie_loss.mean().item())
        all_corrupted_probs.append(corrupted_probs)
# %%

all_clean_probs = torch.cat(all_clean_probs, dim=0)
all_corrupted_probs = torch.cat(all_corrupted_probs, dim=0)
acc = all_clean_probs.mean()
corrupted_acc = all_corrupted_probs.mean()

print(f"Clean {acc}, corrupted {corrupted_acc}")

for layer in aie_probs:
    aie_probs[layer] = torch.cat(aie_probs[layer], dim=0)

# %%
sns.scatterplot(x=all_clean_probs.cpu(), y=all_corrupted_probs.cpu())
plt.show()

# %%

train_split = 0.6
train_line = []
test_line = []
# %%
for layer in aie_probs:
    # sns.histplot(aie_probs[layer].cpu().clamp(min=0,max=1), bins=100)
    test_start = math.ceil(train_split * len(aie_probs[layer]))
    # print(f"Train layer {layer}, AIE", aie_probs[layer][:test_start].mean().item())
    # print(f"Test layer {layer}, AIE", aie_probs[layer][test_start:].mean().item())

    train_line.append((aie_probs[layer][:test_start]).mean().item())
    test_line.append((aie_probs[layer][test_start:]).mean().item())
# %%

sns.lineplot(train_line, label="train")
sns.lineplot(test_line, label="test")

# %%
sns.scatterplot(x=aie_probs[10].cpu(), y=aie_probs[30].cpu())

# %%
torch.save(all_clean_probs, f"{folder}/{node_type}/{node_type}{ablate_type}_clean_probs.pth")
torch.save(all_corrupted_probs, f"{folder}/{node_type}/{node_type}{ablate_type}_corrupted_probs.pth")
torch.save(aie_probs, f"{folder}/{node_type}/{node_type}{ablate_type}_aie.pth")

# %%
node_types = {"attn": "Attention", "mlp": "MLP"}
labels={"": "Optimal ablation", "_gauss": "Gaussian noise"}
for s1 in node_types:
    for s2 in labels:
        aie = torch.load(f"{folder}/{s1}/{s1}{s2}_aie.pth")
        corrupted_probs = torch.load(f"{folder}/{s1}/{s1}{s2}_corrupted_probs.pth")

        aie_means = []
        for i in range(len(aie)):
            aie_means.append(aie[i].mean().item())
        print(labels[s2])
        intv = np.arange(len(aie_means))
        width = 0.5

        ax = plt.bar(intv + (width if s2=="" else 0), aie_means, width, label=labels[s2])
        ax = sns.lineplot(x=[0, len(aie_means)], y=[corrupted_probs.mean().item(), corrupted_probs.mean().item()], color=ax[0].get_facecolor())
    # for i, t in enumerate(ax.get_xticklabels()):
    #     if (i % 5) != 0:
    #         t.set_visible(False)
    ax.set(xlabel="Layer no", ylabel="Probability on correct label")
    ax.set_title(f"Recovered probability on {node_types[s1]} layers")
    plt.legend()
    plt.savefig(f"results/causal_tracing/overall/{s1}.png")
    plt.show()



# %%
