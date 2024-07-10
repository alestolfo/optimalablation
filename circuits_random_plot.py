# %%

import torch
import glob
import os
from itertools import product
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
limits = {
    "ioi": (400,500),
    "gt": (200,300)}

dataset_list = {
    "gt": "Greater-Than", 
    "ioi": "IOI"
}
ablation_types = [
    "mean", "resample", "oa", 
    "cf"]
ax_labels = {
    "mean": "Mean", 
    "resample": "Resample",
    "oa": "Optimal",
    "cf": "Counterfactual"
}
ablation_type="oa"

for (ds, ablation_type) in product(limits, ablation_types):
    print("Ablation type", ablation_type)
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
    
    good_edges = None
    good_loss = 1000 

    if ablation_type == "oa":
        acd_folder = f"results/pruning/{ds}/{ablation_type}/unif/*"
        
        for lamb_folder in glob.glob(acd_folder):
            print(lamb_folder)

            loss_curve_file = f"{lamb_folder}/fit_loss_log.pkl"
            if not os.path.exists(loss_curve_file):
                print("NO LOG FOUND")
                continue

            with open(loss_curve_file, "rb") as f:
                loss_curve = pickle.load(f)
            loss = loss_curve.stat_book['kl_loss'][epochs]

            mask = retrieve_mask(lamb_folder)
            _, _, c_e, _, _ = prune_dangling_edges(discretize_mask(mask, 0))

            if c_e <= max_edges and (good_edges is None or loss < good_loss):
                good_edges = c_e
                good_loss = loss
    else:
        acd_file = f"results/pruning/{ds}/{ablation_type}/unif/post_training.pkl"
        if os.path.exists(acd_file):
            with open(acd_file, "rb") as f:
                losses = pickle.load(f)
            
            for i, c_e in enumerate(losses['clipped_edges']):
                loss = losses['losses'][i]

                if c_e <= max_edges and (good_edges is None or loss < good_loss):
                    good_edges = c_e
                    good_loss = loss
        else:
            print(f"FILE {acd_file} not found")
                
    if good_edges is not None:
        print("UGS loss", good_loss)
        zscore = round((good_loss-mean) / stdev, 2)
        left_align = (max_edges - good_edges) < 32
        print('Z score', zscore)
        # plt.plot(good_edges, good_loss, '*', markersize=10, color="red", label="UGS circuit")
        # plt.text(x=good_edges + (-2 if left_align else 2), y=good_loss, s = f"   Z-score: {zscore}", va="bottom", ha="right" if left_align else "left")
    
    # sns.scatterplot(x=agg_edges, y=agg_losses, label="Random circuits", s=10)
    # plt.title(f"{ax_labels[ablation_type]} ablation on {dataset_list[ds]}, random circuits")
    # plt.savefig(f"{out_folder}/z.png")
    # plt.show()

    # sns.histplot(agg_losses, label="Random circuits")
    # plt.title(f"{ax_labels[ablation_type]} ablation on {dataset_list[ds]}, random circuits")
    # plt.savefig(f"{out_folder}/dist.png")
    # plt.show()
# %%

