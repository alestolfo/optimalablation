# %%

import torch
import glob
import os
import numpy as np
import pickle
import seaborn as sns
from utils.training_utils import LinePlot
import matplotlib.pyplot as plt
from utils.circuit_utils import retrieve_mask, discretize_mask, prune_dangling_edges

# %%
sns.set()

# %%
epochs = 200
limits = {"ioi": (400,500), "gt": (200,300)}

dataset_list = {"gt": "Greater-Than", "ioi": "IOI"}
ablation_types = ["mean", "resample", "oa", "cf"]
ax_labels = {
    "mean": "Mean", 
    "resample": "Resample",
    "oa": "Optimal",
    "cf": "Counterfactual"
}
ablation_type="oa"


def retrieve_data_point(lamb_folder, min_edges, max_edges):
    loss_curve_file = f"{lamb_folder}/fit_loss_log.pkl"
    if not os.path.exists(loss_curve_file):
        return False

    with open(loss_curve_file, "rb") as f:
        loss_curve = pickle.load(f)

    mask = retrieve_mask(lamb_folder)
    _, _, c_e, _, _ = prune_dangling_edges(discretize_mask(mask, 0))

    if c_e >= min_edges and c_e <= max_edges:
        good_loss = loss_curve.stat_book['kl_loss'][epochs]
        zscore = round((good_loss-mean) / stdev, 2)
        left_align = (max_edges - c_e) < 32
        plt.plot(c_e, good_loss, '*', markersize=10, color="red", label="UGS circuit")
        plt.text(x=c_e + (-2 if left_align else 2), y=good_loss, s = f"   Z-score: {zscore}", va="bottom", ha="right" if left_align else "left")

        print("ACD loss", good_loss)
        return True
    return c_e

for ds in limits:
    out_folder = f"results/pruning_random/{ds}/{ablation_type}"
    min_edges, max_edges = limits[ds]

    agg_losses = []
    agg_edges = []
    folder = f"{out_folder}/*.pth"
    for g in glob.glob(folder):
        losses = torch.load(g)
        agg_losses += losses['loss']
        agg_edges += losses['edges']

    mean = np.mean(agg_losses)
    stdev = np.std(agg_losses)

    print("Mean loss", mean)
    print("Std loss", stdev)
    
    acd_folder = f"results/pruning/{ds}/{ablation_type}/unif/*"
    
    closest_e = 0
    closest_lamb_folder = None
    plotted = False

    for lamb_folder in glob.glob(acd_folder):
        print(lamb_folder)
        c_e = retrieve_data_point(lamb_folder, min_edges, max_edges)
        if c_e is True:
            plotted = True
            break
        elif c_e is not False and c_e >= closest_e and c_e < min_edges:
            closest_e = c_e
            closest_lamb_folder = lamb_folder
    
    if not plotted:
        retrieve_data_point(closest_lamb_folder, 0, max_edges)
    
    sns.scatterplot(x=agg_edges, y=agg_losses, label="Random circuits", s=10)
    plt.title(f"{ax_labels[ablation_type]} ablation on {dataset_list[ds]}, random circuits")
    plt.savefig(f"{out_folder}/z.png")
    plt.show()

    sns.histplot(agg_losses, label="Random circuits")
    plt.title(f"{ax_labels[ablation_type]} ablation on {dataset_list[ds]}, random circuits")
    plt.savefig(f"{out_folder}/dist.png")
    plt.show()
# %%

