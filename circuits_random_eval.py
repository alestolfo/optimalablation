# %%

import torch
import glob
import os
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
ablation_type="resample"

for ds in limits:
    min_edges, max_edges = limits[ds]

    agg_losses = []
    agg_edges = []
    folder = f"results/pruning_random/{ds}/{ablation_type}/*"
    for g in glob.glob(folder):
        losses = torch.load(g)
        agg_losses += losses['loss']
        agg_edges += losses['edges']
    
    acd_folder = f"results/pruning/{ds}/oa/unif/*"
    for lamb_folder in glob.glob(acd_folder):
        print(lamb_folder)
        loss_curve_file = f"{lamb_folder}/fit_loss_log.pkl"
        if not os.path.exists(loss_curve_file):
            continue

        with open(loss_curve_file, "rb") as f:
            loss_curve = pickle.load(f)

        mask = retrieve_mask(lamb_folder)
        _, _, c_e, _, _ = prune_dangling_edges(discretize_mask(mask, 0))

        if c_e >= min_edges and c_e <= max_edges:
            plt.plot(c_e, loss_curve.stat_book['kl_loss'][epochs], '*', markersize=10, color="red", label="UGS circuit")

    sns.scatterplot(x=agg_edges, y=agg_losses, label="Random circuits", s=10)
    plt.title(ds)
    plt.show()

    sns.histplot(agg_losses, label="Random circuits")
    plt.title(ds)
    plt.show()
# %%

