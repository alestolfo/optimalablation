# %%
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import glob
import os

sns.set()

# %%
def plot_points(k, x):
    print(x)
    if os.path.exists(f"{x}/post_training.pkl"):
        with open(f"{x}/post_training.pkl", "rb") as f:
            log = pickle.load(f)
        # print(log)
        print(log['tau'])
        sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label=f"{k} post training", marker="X", s=50)

def plot_pareto(folder):
    folder, manual_folder = folder

    plt.figure(figsize=(15,30))

    for k, x in folder.items():
        ax = None
        color = None
        print(x)
        for path in glob.glob(f"{x}/report/*"):
            # out_path=f"pruning_edges_auto/report/ioi_zero_init_{str(reg_lamb).replace('.', '-')}.pkl"
            with open(path, "rb") as f:
                log = pickle.load(f)
            # print(log)
            lamb = path.split("/")[-1].replace(".pkl","")
            print(lamb)
            # if k == "vertex":
            #     print(log['tau'])
            if ax is None:
                ax = sns.lineplot(x=log["clipped_edges"], y=log["losses"], label=k)
                color = ax.lines[-1].get_color()
            else:
                sns.lineplot(x=log["clipped_edges"], y=log["losses"], ax=ax, color=color)
            
            # print(log['tau'])
            undot = True
            for i,tau in enumerate(log['tau']):
                if tau >= -0.5 and undot:
                    plt.plot(log["clipped_edges"][i], log["losses"][i], 'k^')
                    undot = False
                if tau >= 1:
                    plt.plot(log["clipped_edges"][i], log["losses"][i], 'ks')
                    break
        plot_points(k, x)
    
    for k, x in manual_folder.items():
        plot_points(k,x)
        # if os.path.exists(f"{x}/pre_training.pkl"):
        #     with open(f"{x}/pre_training.pkl", "rb") as f:
        #         log = pickle.load(f)
        #     print(log)
        #     sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label="pre training", marker="X", s=50)
    
    plt.ylim(0,0.3)
    plt.xlim(0,4000)
    plt.gca().xaxis.set_major_locator(MultipleLocator(200)) # x gridlines every 0.5 units
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01)) # y gridlines every 0.5 units

    plt.show()

# %%
ax = None
# reg_lambs = [2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
folders=[
    ({
        "edges": "pruning_edges_auto/ioi_clipped_edges", 
        "vertex": "pruning_vertices_auto/ioi_with_mlp", 
        "edges from vertex prior": "pruning_edges_auto/ioi_vertex_prior"
        # "reset_optim": "pruning_edges_auto/ioi_reinit",  
        # "prune_retrain": "pruning_edges_auto/ioi_reinit_lr",
    }, {
        "ACDC": "acdc_ioi_runs",
        "iterative": "pruning_edges_auto/ioi_iter",
        "manual": "pruning_vertices_auto/ioi_manual",
    }),
    # ([], ["pruning_edges_auto/ioi_iter"]),
    # "pruning_edges_auto-2-24/ioi-2-26",
    # "pruning_edges_auto-2-24/gt",
    # "pruning_edges_auto-2-26/ioi_zero_init",
]

for folder in folders:
    plot_pareto(folder)


# %%

# plt.plot(1176, 0.09452762454748154, 'gs')
# plt.plot(1256, 0.10203401073813438, 'gs')
# plt.plot(662, 0.10643498972058296, 'go')
# plt.plot(2218, 0.10283389091491699, 'gs')
# plt.plot(2896, 0.09150597080588341, 'go')
# # plt.plot(1041, 0.06662799082696438, 'rP')
# plt.plot(1666, 0.06662799082696438, 'rP')
# plt.plot(1644, 0.09744843393564225, 'go')

# manual 1041, 0.06662799082696438

# 5e-4 -> -1, 662, 0.10643498972058296
# 2e-4 -> 1.5, 1176, 0.09783206954598427
# 2e-4 -> 1, 1256, 0.10203401073813438

# %%

# ax = None
# reg_lambs = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
# for reg_lamb in reg_lambs:
#     out_path=f"pruning_edges_auto/report/gt_{str(reg_lamb).replace('.', '-')}.pkl"
#     with open(out_path, "rb") as f:
#         log = pickle.load(f)
#     if ax is None:
#         sns.lineplot(x=log["clipped_edges"], y=log["losses"], label=reg_lamb)
#     else:
#         sns.lineplot(x=log["clipped_edges"], y=log["losses"], ax=ax, label=reg_lamb)
    
#     print(log['tau'])
#     undot = True
#     for i,tau in enumerate(log['tau']):
#         if tau >= -1 and undot:
#             plt.plot(log["clipped_edges"][i], log["losses"][i], 'k^')
#             undot = False
#         if tau >= 1:
#             plt.plot(log["clipped_edges"][i], log["losses"][i], 'ks')
#             break

# plt.plot(262, 0.035, 'rP')

# %%
