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
import matplotlib.ticker as plticker
import time
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# %%
# with open("pruning/pruning_modes_ioi/modes_1.pkl", "rb") as f:
#     # n_layers x n_heads x d_model
#     modal_values = pickle.load(f)

pruning_values = {}
# with open("pruning/pruning_outputs/ioi_spec_modes/train_823.pkl", "rb") as f:
with open("pruning/pruning_modes_ioi_missing_modes/train_271.pkl", "rb") as f:
#     # n_layers x n_heads x d_model
    pruning_values["joint_modes"] = pickle.load(f)
with open("pruning/pruning_means_ioi/train_251.pkl", "rb") as f:
#     # n_layers x n_heads x d_model
    pruning_values["means_by_pos"] = pickle.load(f)

# %%
MANUAL_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}

positives = {x for y in MANUAL_CIRCUIT for x in MANUAL_CIRCUIT[y]}

# %%

for lab in ["joint_modes", "means_by_pos"]:
    alpha_ranking = torch.stack(pruning_values[lab],dim=0)[:,:,0]
    v, i = torch.topk(alpha_ranking.flatten(), 144, sorted=True)
    indices = np.array(np.unravel_index(i.cpu().detach().numpy(), alpha_ranking.shape)).T
    ROC_graph_x = []
    ROC_graph_y = []
    pos_count = 0
    neg_count = 0
    for i in range(indices.shape[0]):
        if (indices[i][0],indices[i][1]) in positives:
            pos_count += 1 / len(positives)
        else:
            neg_count += 1 / (144 - len(positives))
        ROC_graph_x.append(neg_count)
        ROC_graph_y.append(pos_count)

    ax = sns.lineplot(x=ROC_graph_x, y=ROC_graph_y, estimator=None, label=lab)

loc = plticker.MultipleLocator(base=0.25)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

plt.title("ROC curves")
plt.grid(visible=True)
plt.xlabel("False positives")
plt.ylabel("True positives")

# %%
