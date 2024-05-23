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
from utils.tracing_utils import get_subject_tokens, replace_subject_tokens, gauss_subject_tokens, patch_component_token_pos, patch_component_last_token, patch_component_all_tokens
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
token_type = "last"
# node_type="mlp"
# ablate_type="gauss"
train_split = 0.6

for token_type, node_type, ablate_type in [
    (x,y,z) for x in ["last", "last_subject"] for y in ["attn", "mlp"] for z in ["oa", "gauss"]
]:
    with open(f"{ds_path}/{ds_name}.pkl", 'rb') as f:
        correct_prompts = pickle.load(f)
    test_start = math.ceil(train_split * len(correct_prompts))
    correct_prompts = correct_prompts[test_start:]

    data_loader = DataLoader(correct_prompts, batch_size=batch_size, shuffle=True)
    data_iter = iter(data_loader)
    causal_layers = [i for i in range(0,48)]
    ct_layers = len(causal_layers)

    # init_token = torch.randn((ct_layers + 1, d_model)).to(device)

    gauss = ablate_type == "gauss"
    with open(f"{folder}/fact_{token_type}_{node_type}_null_tokens_{causal_layers[0]}_{causal_layers[-1]}.pkl", "rb") as f:
        null_token = pickle.load(f).to(device)
    # init_token = torch.load(f"{folder}/subject_means.pth")
    # init_token = init_token.unsqueeze(0).repeat(ct_layers+1, 1)

    attn_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out" 
    mlp_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out" 
    covs = torch.load(f"{folder}/stds.pth")
    stds = covs.diag().sqrt() * 3

    aie_probs = {}
    for layer in causal_layers:
        aie_probs[layer] = []
    all_clean_probs = []
    all_corrupted_probs = []
    lp = LinePlot(["kl_loss", "step_sz"])

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_iter)):

            n_tries = 1 if node_type == "oa" else 5

            tokens, subject_pos = get_subject_tokens(batch, tokenizer, mode=mode)
            bsz = tokens.shape[0]

            tokens = tokens.repeat(ct_layers + 2, 1).to(device)

            if token_type == "last":
                patch_token_pos = None
            elif token_type == "last_subject":
                mask = torch.zeros_like(tokens).to(device)
                mask[subject_pos[:,0], subject_pos[:,1]] = 1
                last_subject_pos = (mask * torch.arange(mask.shape[-1]).to(device)).argmax(dim=-1)
                patch_token_pos = torch.stack([torch.arange(mask.shape[0]).to(device), last_subject_pos], dim=-1)
            elif token_type == "all_subject":
                patch_token_pos = subject_pos

            for k in range(n_tries):
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
                            partial(patch_component_last_token, bsz, j) if token_type == "last"
                            else partial(patch_component_token_pos, bsz, j, patch_token_pos)) 
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
                all_corrupted_probs.append(torch.minimum(corrupted_probs, probs))

                for j, layer in enumerate(causal_layers):
                    layer_probs = layer_results[bij, j, target_tokens].exp()
                    aie_probs[layer].append(torch.minimum(layer_probs, probs))

    all_clean_probs = torch.cat(all_clean_probs, dim=0)
    all_corrupted_probs = torch.cat(all_corrupted_probs, dim=0)

    sns.scatterplot(x=all_clean_probs.cpu(), y=all_corrupted_probs.cpu(), s=5)
    acc = all_clean_probs.mean()
    corrupted_acc = all_corrupted_probs.mean()

    # print(f"Clean {acc}, corrupted {corrupted_acc}")
    for layer in aie_probs:
        aie_probs[layer] = torch.cat(aie_probs[layer], dim=0)

    # for layer in aie_probs:
    #     # sns.histplot(aie_probs[layer].cpu().clamp(min=0,max=1), bins=100)
    #     print(f"Layer {layer}, AIE", aie_probs[layer].mean().item())

    torch.save(all_clean_probs, f"{folder}/{node_type}/{token_type}_{ablate_type}_clean_probs.pth")
    torch.save(all_corrupted_probs, f"{folder}/{node_type}/{token_type}_{ablate_type}_corrupted_probs.pth")
    torch.save(aie_probs, f"{folder}/{node_type}/{token_type}_{ablate_type}_aie.pth")

# %%
test_start = math.ceil(0.6 * len(correct_prompts))
node_types = {"attn": "Attention", "mlp": "MLP"}
labels={"oa": "Optimal ablation", "gauss": "Gaussian noise"}
token_types={"last": "last token", "last_subject": "last subject token", "all_subject": "all subject tokens"}
for s3 in token_types:
    for s1 in node_types:
        for s2 in labels:
            aie = torch.load(f"{folder}/{s1}/{s3}_{s2}_aie.pth")
            corrupted_probs = torch.load(f"{folder}/{s1}/{s3}_{s2}_corrupted_probs.pth")

            corrupted_probs = corrupted_probs[test_start:]

            aie_means = []
            aie_stds = []
            n_samples = aie[0].nelement()
            for i in range(len(aie)):
                aie[i] = aie[i][test_start:]
                aie_means.append(aie[i].mean().item() - corrupted_probs.mean().item())
                aie_stds.append(aie[i].std().item())
            print(labels[s2])
            intv = np.arange(len(aie_means))
            aie_means = np.array(aie_means)
            aie_stds = np.array(aie_stds)

            width = 0.5
            ax = plt.bar(intv + (width if s2=="oa" else 0), aie_means, 0.5, label=labels[s2])

            # lcb = sns.lineplot(aie_means - 1.96 * aie_stds / math.sqrt(n_samples), color=ax[0].get_facecolor())
            # ucb = sns.lineplot(aie_means + 1.96 * aie_stds / math.sqrt(n_samples), color=ax[0].get_facecolor())
            ax = sns.lineplot(x=[0, len(aie_means)], y=[0, 0], color="black")
            # for i, t in enumerate(ax.get_xticklabels()):
            #     if (i % 5) != 0:
            #         t.set_visible(False)

        ax.set(xlabel="Layer number", ylabel="Probability on correct label")
        ax.set_title(f"Recovered probability on {node_types[s1]} layers patching {token_types[token_type]}", fontsize=20)
        plt.legend()
        plt.savefig(f"results/causal_tracing/overall/{s1}.png")
        plt.show()
    break
        


# %%
