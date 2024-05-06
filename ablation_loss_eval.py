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
import os
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, load_args, update_means_variances, update_means_variances_mixed, update_means_variances_exponential, plot_no_outliers
from utils.MaskConfig import VertexInferenceConfig
from utils.task_datasets import get_task_ds
from pruners.VertexPruner import VertexPruner
from mask_samplers.AblationMaskSampler import SingleComponentMaskSampler
from utils.circuit_utils import edges_to_mask, mask_to_nodes, prune_dangling_edges

# %%
folder="results/ablation_loss"
dataset_list = ["gt", "ioi"]
ablation_types = ["zero", "mean", "resample", "refmean", "oca", "cf"]

ablation_data = {}
for ds in dataset_list:
    ablation_data[ds] = {}
    for ablation_type in ablation_types:
        ablation_data[ds][ablation_type] = torch.load(f"{folder}/{ds}/{ablation_type}_results.pth")

# %%

n = len(ablation_types)

for ds in dataset_list:
    f, axes = plt.subplots(n, n, figsize=(5*n, 5*n))
    for i in range(n):
        x = ablation_types[i]
        for j in range(i+1, n):
            y = ablation_types[j]
            plot_no_outliers(sns.scatterplot, .03, 
                            ablation_data[ds][x]['head_losses'].log(), 
                            ablation_data[ds][y]['head_losses'].log(),
                            axes[i,j], xy_line=True, args={"x": x, "y": y, "s": 10, "corr": True})
            plot_no_outliers(sns.scatterplot, 0, 
                             # ranks
                            (ablation_data[ds][x]['head_losses'] > ablation_data[ds][x]['head_losses'].squeeze(-1)).sum(dim=-1), 
                            (ablation_data[ds][y]['head_losses'] > ablation_data[ds][y]['head_losses'].squeeze(-1)).sum(dim=-1),
                            axes[j,i], xy_line=True, args={"x": x, "y": y, "s": 10, "corr": True})
    plt.savefig(f"{folder}/{ds}.png")
    plt.show()


# %%

n_layers = 12
n_heads = 12
edges_per_node = {'attn': 146, 'mlp': 157}

for ds in dataset_list:
    edge_list = torch.load(f"results/pruning_edges_auto/{ds}_acdc/edges_manual.pth")
    edge_mask = edges_to_mask(edge_list)
    _, _, _, attn_nodes, mlp_nodes = prune_dangling_edges(edge_mask)
    node_list = {'attn': attn_nodes.squeeze(0).nonzero().tolist(), 'mlp': mlp_nodes.nonzero().flatten().tolist()}

    idx_dict = {}

    
    for layer_no, head_no in node_list['attn']:
        idx_dict[layer_no * n_heads + head_no] = (attn_nodes[0, layer_no, head_no].item(), edges_per_node['attn'])
    
    for mlp_no in node_list['mlp']:
        idx_dict[n_layers * n_heads + mlp_no] = (mlp_nodes[0, mlp_no].item(), edges_per_node['mlp'])
        
    for ablation_type in ablation_types:
        true_positives = [0]
        false_positives = [0]

        for node_idx in ablation_data[ds][ablation_type]['head_losses'].argsort(dim=0, descending=True).flatten().tolist():
            if node_idx in idx_dict:
                # prop = idx_dict[node_idx][0] / idx_dict[node_idx][1]
                prop = 1
                true_positives.append(true_positives[-1] + prop)
                false_positives.append(false_positives[-1] + 1 - prop)
            else:
                true_positives.append(true_positives[-1])
                false_positives.append(false_positives[-1] + 1)
        
        sns.lineplot(x=false_positives, y=true_positives, label=ablation_type, estimator=None)
    plt.savefig(f"{folder}/{ds}_roc_nodes.png")
    plt.show()

# %%
