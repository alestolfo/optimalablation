# %%

import torch
import glob
import os
import seaborn as sns
from task_datasets import IOIConfig
import matplotlib.pyplot as plt

sns.set()

# %%

folders = {"acdc": "acdc_ioi_runs", "edges": "pruning_edges_auto/ioi_clipped_edges", "vertex_prior": "pruning_edges_auto/ioi_vertex_prior", "vertices": "pruning_vertices_auto/ioi_with_mlp"}

fitted_modes = {}
prefitted_modes = {}

# %%

for f_type, folder in folders.items():
    print(folder)
    fitted_modes[f_type] = {}
    prefitted_modes[f_type] = {}
    for lamb_path in glob.glob(f"{folder}/*"):
        lamb = lamb_path.split("/")[-1]
        if lamb != "manual":
            try:
                float(lamb[0])
                float(lamb[-1])
            except:
                continue
        print(lamb)
        fitted_modes[f_type][lamb] = {}
        if os.path.exists(f"{lamb_path}/snapshot.pth"):
            # print("loading prefitted modes")
            param_dict = torch.load(f"{lamb_path}/snapshot.pth")["pruner_dict"]
            prefitted_modes[f_type][lamb] = {"modal_attention": param_dict["modal_attention"], "modal_mlp": param_dict["modal_mlp"]}
        
        fitted_files = glob.glob(f"{lamb_path}/fit_modes_*.pth")
        for f in fitted_files:
            tau = f.split("/")[-1].replace("fit_modes_", "").replace(".pth", "")
            fitted_modes[f_type][lamb][tau] = torch.load(f)

# %%
            
def compute_similarity(attn_1, attn_2, plot=False):
    output = (attn_1['modal_attention']-attn_2['modal_attention']).norm(dim=-1)

    if plot:
        sns.histplot(output.detach().flatten().cpu().numpy())

    return output


# for x in prefitted_modes:
#     for y in prefitted_modes[x]:
#         print(prefitted_modes[x][y]['modal_attention'].shape)
# %%

for x in prefitted_modes:
    print(x)
    for y in prefitted_modes[x]:
        print(y)
        if '0.0' in fitted_modes[x][y]:
            compute_similarity(
                prefitted_modes[x][y], fitted_modes[x][y]['0.0'], plot=True
            )
            # compute_similarity(
            #     prefitted_modes[x][y], {"modal_attention": means[0]}, plot=True
            # )
        elif '-1.0' in fitted_modes[x][y]:
            compute_similarity(
                prefitted_modes[x][y], fitted_modes[x][y]['-1.0'], plot=True
            )
            # compute_similarity(
            #     prefitted_modes[x][y], {"modal_attention": means[0]}, plot=True
            # )
        plt.show()
# %%


config = IOIConfig(20, "cuda:0")
# %%
means = config.init_modes()
# %%
means[0].shape
# %%
