# %%
import torch
import numpy as np 
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import plot_no_outliers
from utils.circuit_utils import edges_to_mask, prune_dangling_edges

# %%
sns.set()

folder="results/ablation_loss"
plot_folder="plots_export/ablation_loss"
dataset_list = {
    # "gt": "Greater-Than", 
    "ioi": "IOI"}
ablation_types = ["zero", "mean", "resample", "cf_mean", "oa_specific", "cf"]
ax_labels = {
    "zero": "Zero",
    "mean": "Mean", 
    "resample": "Resample",
    "cf_mean": "CF-Mean",
    "oa_specific": "Optimal",
    "cf": "CF"
}

if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

ablation_data = {}
for ds in dataset_list:
    ablation_data[ds] = {}
    for ablation_type in ablation_types:
        ablation_data[ds][ablation_type] = torch.load(f"{folder}/{ds}/{ablation_type}_results.pth")

# %%

CORR_SIZE = 32
SMALL_SIZE = 12
MEDIUM_SIZE = 32
BIGGER_SIZE = 48

plt.rc('font', size=CORR_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

n = len(ablation_types)

for ds in dataset_list:
    f, axes = plt.subplots(n, n, figsize=(5*n, 5*n))
    for i in range(n):
        x = ablation_types[i]
        sns.histplot(ablation_data[ds][x]['head_losses'].log().cpu(), ax=axes[i,i], legend=False)
        axes[i,i].set(xlabel=ax_labels[x])
        for j in range(i+1, n):
            y = ablation_types[j]
            plot_no_outliers(sns.scatterplot, .02, 
                            ablation_data[ds][x]['head_losses'].log(), 
                            ablation_data[ds][y]['head_losses'].log(),
                            axes[i,j], xy_line=True, args={"x": ax_labels[x], "y": ax_labels[y], "s": 20, "corr": True})
            plot_no_outliers(sns.scatterplot, 0, 
                             # ranks
                            (ablation_data[ds][x]['head_losses'] > ablation_data[ds][x]['head_losses'].squeeze(-1)).sum(dim=-1), 
                            (ablation_data[ds][y]['head_losses'] > ablation_data[ds][y]['head_losses'].squeeze(-1)).sum(dim=-1),
                            axes[j,i], xy_line=True, args={"x": ax_labels[x], "y": ax_labels[y], "s": 20, "corr": True})
    plt.suptitle(f"Correlation plots of ablation loss measurements on {dataset_list[ds]}")
    plt.tight_layout()
    plt.subplots_adjust(top=.96)
    plt.savefig(f"{plot_folder}/{ds}.png")
    plt.show()

# %%

# comparison table

for ds in dataset_list:
    for i in range(n):
        x = ablation_types[i]
        ratios = ablation_data[ds]['oa_specific']['head_losses'] / ablation_data[ds][x]['head_losses']
        if x != "cf":
            ratios = ratios.clamp(max=1)
        print(x, ratios.mean().item())

# %%

for ds in dataset_list:
    for i in range(n):
        x = ablation_types[i]
        print(x, ablation_data[ds][x]['head_losses'].mean().item())