# %%
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import glob
import os

sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
# plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})


# %%
def plot_points(k, x, color=None):
    print(x)
    if os.path.exists(f"{x}/post_training.pkl"):
        with open(f"{x}/post_training.pkl", "rb") as f:
            log = pickle.load(f)
        # print(log)
        print(log['tau'])
        if color is not None:
            ax = sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label=f"{k} post training", marker="X", s=50, color=color)
        else:
            ax = sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label=f"{k} post training", marker="X", s=50)

        for i,t in enumerate(log['tau']):
            print(t, log["lamb"][i], log['clipped_edges'][i], log['losses'][i])
            if log["lamb"][i] == "manual":
                plt.plot(log["clipped_edges"][i], log["losses"][i], 'k*', markersize=10)
    return ax

def plot_pareto(pms):
    folder, manual_folder, y_bound, x_bound, task_name = pms

    fig = plt.figure(figsize=(10,15))

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
        if k == "ACDC":
            ax = plot_points(k,x, color="black")
        else:
            ax = plot_points(k,x)
        if os.path.exists(f"{x}/pre_training.pkl"):
            with open(f"{x}/pre_training.pkl", "rb") as f:
                log = pickle.load(f)
            print(log)
            sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label="pre training", marker="X", s=50)
    
    plt.ylim(0,y_bound)
    plt.xlim(0,x_bound)
    plt.gca().xaxis.set_major_locator(MultipleLocator(200)) # x gridlines every 0.5 units
    plt.minorticks_on()
    plt.tick_params(which='minor', bottom=False, left=False)

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2)) # x gridlines every 0.5 units
    plt.grid(visible=True, which='minor', color='w', linewidth=0.5)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01)) # y gridlines every 0.5 units
    plt.xlabel("Edges kept")
    plt.ylabel("KL divergence")
    plt.savefig(f"pareto/{task_name}.png")
    plt.show()

# %%
ax = None
# reg_lambs = [2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
folders=[
    ({
        "vertex": "results/pruning_vertices_auto/ioi_with_mlp", 
        "edges HC": "results/pruning_edges_auto/ioi_edges", 
        "edges HC (vertex prior)": "results/pruning_edges_auto/ioi_vertex_prior", 
        "edges uniform": "results/pruning_edges_auto/ioi_edges_unif", 
    }, {
        "ACDC": "results_baseline/acdc_ioi_runs",
        "eap": "results_baseline/eap_ioi_runs"
    }, 0.15, 3000, "ioi"),
    ({
        "vertex": "results/pruning_vertices_auto/gt", 
        "edges HC": "results/pruning_edges_auto/gt_edges", 
        "edges HC (vertex prior)": "results/pruning_edges_auto/gt_vertex_prior", 
        "edges uniform": "results/pruning_edges_auto/gt_edges_unif", 
    }, {
        "ACDC": "results_baseline/acdc_gt_runs",
        "eap": "results_baseline/eap_gt_runs"
    }, 0.05,1000,"gt"),
]

for folder in folders:
    plot_pareto(folder)
