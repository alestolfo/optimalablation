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

means_only=False
include_mlp=True
means_by_seq_pos=True
init_modes_path = f"oca/owt/means_attention.pkl"
folder = "oca/ioi"
means_path = f"{folder}/means_attention.pkl"
modes_path = f"{folder}/modes.pkl"
baseline_path = f"{folder}/baseline_resid.pkl"

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 6
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.eval()
# model.cfg.use_attn_result = True

# %%
task_ds = OWTConfig(owt_iter, device)
# task_ds = IOIConfig(batch_size, device)
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
resid_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_post"
final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

# %%

with open(f"{folder}/baseline_resid.pkl", "rb") as f:
    baseline_dist = pickle.load(f)

print(baseline_dist.shape)

# %%

eigenvalues = []
components = []
baseline_pca = []

# 3 metrics: KDE, nearest-neighbors, PCA
for i in tqdm(range(n_layers)):
    u, s, v = torch.pca_lowrank(baseline_dist[:,i],d_model)
    eigenvalues.append(s)
    components.append(v)
    baseline_pca.append((u/s).abs().sum(dim=1))

eigenvalues = torch.stack(eigenvalues, dim=0)
components = torch.stack(components, dim=0)
baseline_pca = torch.stack(baseline_pca, dim=0)

# %%

baseline_dist = baseline_dist[:10000]
torch.cuda.empty_cache()

# comp_baseline_dist = baseline_dist.unsqueeze(1)

# %%

# min_vals = baseline_dist.min(dim=0)[0].min(dim=-1)[0]
# max_vals = baseline_dist.max(dim=0)[0].min(dim=-1)[0]

# # %%

# def quantize_tensor(ts, min_val, max_val):
#     scale = (max_val - min_val) / 255
#     quantized_tensor = torch.clamp(((ts - min_val) / scale).round(), 0, 255).byte()
#     return quantized_tensor

# # %%
# qbaseline_dist = []

# for i in range(n_layers):
#     qbaseline_dist.append(quantize_tensor(baseline_dist[:,i], min_vals[i], max_vals[i]))

# qbaseline_dist = torch.stack(qbaseline_dist, dim=1)

# %%

dist_threshold = []

for j in range(n_layers):
    dist_threshold.append([])
    for i in tqdm(range(baseline_dist.shape[0])):
        dist_thresh = torch.topk((baseline_dist[:,j]-baseline_dist[i,j]).norm(dim=-1), 100)[0][-1]
        dist_threshold[j].append(dist_thresh)

dist_threshold = torch.tensor(dist_threshold).to(device)

# %%
torch.save({"eigenvalues": eigenvalues, "components": components, "baseline_pca": baseline_pca, "dist_threshold": dist_threshold.permute(1,0)}, f"{folder}/baseline_stats.pth")

# %%
obj = torch.load(f"{folder}/baseline_stats.pth")
    
def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def mean_ablation_hook_attention_all_tokens(constants, bsz, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(n_heads):
        # print(attentions.shape)
        # print(torch.cat([constants[:-1,i], constants[[-1],i].repeat(attentions.shape[1]-constants.shape[0],1)],dim=0).clone().shape)
        attentions[-start:-start+bsz,:,i] = torch.cat([constants[:-1,i], constants[[-1],i].repeat(1+attentions.shape[1]-constants.shape[0],1)],dim=0).clone()
        start -= bsz
    return attentions

def resample_ablation_hook_attention_all_tokens(bsz, attentions, hook):
    # n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(n_heads):
        # print(attentions.shape)
        # print(torch.cat([constants[:-1,i], constants[[-1],i].repeat(attentions.shape[1]-constants.shape[0],1)],dim=0).clone().shape)
        permutation = torch.randperm(bsz) - start
        # print(attentions[permutation,:,(torch.ones(bsz) * i).int()].shape)
        # print(i)
        # print(attentions[-start:(attentions.shape[0] if -start+bsz == 0 else -start + bsz),:,i].shape)
        # print(start)
        # print(attentions.shape[0])
        attentions[-start:(attentions.shape[0] if -start+bsz == 0 else -start + bsz),:,i] = attentions[permutation,:,(torch.ones(bsz) * i).int()]
        start -= bsz
    return attentions

def mode_ablation_hook_attention_all_tokens(constants, bsz, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    attentions[-start:] = 100
    # print(start)
    # for i in range(n_heads):
    #     attentions[-start:(attentions.shape[0] if -start+bsz == 0 else -start + bsz)] = 100
    #     start -= bsz
    # print(start)
    # print(attentions.shape)
    # attentions[-start:] = 100
    return attentions

def ood_hook_last_token(ood_metric_storage, last_token_mask, layer_no, resid, hook):
    act = resid.unflatten(0, (-1, batch_size))
    act = (act * last_token_mask.unsqueeze(-1)).sum(dim=2)

    # KDE
    flat_act = act.flatten(start_dim=0, end_dim=1)
    dist_to_baseline = (baseline_dist[layer_no] - flat_act.unsqueeze(1)).norm(dim=-1)
    kde = (dist_to_baseline < 40+20 * layer_no).sum(dim=1)

    # NN
    nn = (dist_to_baseline < obj['dist_threshold'][layer_no]).sum(dim=1)

    # PCA
    pca = (einsum("batch d_model, d_model component_no -> batch component_no", flat_act, obj['components'][layer_no]) / obj['eigenvalues'][layer_no]).abs().sum(dim=-1)
    ood_metric_storage.append({"kde": kde, "nn": nn, "pca": pca})

def final_hook_all_tokens(last_token_mask, orig_in, hook):
    out = orig_in.unflatten(0, (-1, batch_size))
    out = (out * last_token_mask.unsqueeze(-1)).sum(dim=2)
    return out


# %%

with open(f"{folder}/means_attention.pkl", "rb") as f:
    means = pickle.load(f)
with open(f"{folder}/modes.pkl", "rb") as f:
    modes = pickle.load(f)

# %%

all_metric_storage = None    
all_kl = None

for i in tqdm(range(60)):
    # modify depending on the dataset

    batch, last_token_pos = task_ds.next_batch()
        
    with torch.no_grad():
        last_token_mask = torch.zeros_like(batch).to(device)
        last_token_mask[torch.arange(last_token_mask.shape[0]), last_token_pos] = 1

        metric_storage = []

        model_results = model.run_with_hooks(
                batch,
                fwd_hooks=[
                    *[(partial(attention_points_filter, layer_no), 
                    partial(mode_ablation_hook_attention_all_tokens,
                                modes[layer_no],
                                batch_size)
                        ) for layer_no in range(n_layers)],
                    *[(partial(resid_points_filter, layer_no), 
                    partial(ablation_hook_copy_all_tokens,
                            batch_size,
                            n_heads)
                        ) for layer_no in range(n_layers)],
                    # *[(partial(resid_post_filter, layer_no), 
                    # partial(ood_hook_last_token,
                    #         metric_storage,
                    #         last_token_mask,
                    #         layer_no)
                    #     ) for layer_no in range(n_layers)],
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

        print(kl_losses.mean())
        sns.histplot(kl_losses.flatten().detach().cpu())
        plt.show()
        if all_metric_storage is None:
            all_metric_storage = metric_storage
            all_kl = kl_losses
        else:
            for i,objc in enumerate(all_metric_storage):
                for k in objc:
                    objc[k] = torch.cat([objc[k], metric_storage[i][k]], dim=0)
            all_kl = torch.cat([all_kl, kl_losses], dim=1)

# %%
for i,objc in enumerate(all_metric_storage):
    for k in objc:
        sns.histplot(objc[k].flatten().cpu())
        plt.savefig(f"oca/oca_result/in_dist_{i}_{k}.png")
        plt.close()
# %%

in_dist_metric_storage = all_metric_storage
# %%
mean_metric_storage = all_metric_storage
# %%
resample_metric_storage = all_metric_storage
# %%

mode_metric_storage = all_metric_storage

# %%

for i,objc in enumerate(all_metric_storage):
    for k in objc:
        for m_storage in [in_dist_metric_storage,mean_metric_storage, resample_metric_storage, mode_metric_storage]:
            sns.histplot(m_storage[i][k].flatten().cpu())
        plt.savefig(f"oca/oca_result/ood_{i}_{k}.png")
        plt.close()

# %%
