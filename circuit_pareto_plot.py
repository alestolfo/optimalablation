# %%
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
import glob
import os
import math
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
# plt.rcParams.update({"xtick.bottom" : True, "ytick.left" : True})

# %%

CORR_SIZE = 20
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=CORR_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plot_folder="plots_export/pareto"
task_lookup = {"ioi": "IOI", "gt": "Greater-Than"}
ablation_lookup = {"mean": "mean", "cf": "counterfactual", "resample": "resample", "oa": "optimal"}

# %%

def plot_points(k, log_file, color=None, manual_only=False):
    with open(log_file, "rb") as f:
        log = pickle.load(f)
    # print(log)

    for i, lamb in enumerate(log['lamb']):
        if lamb == "manual":
            manual_run = {}
            for ke in log:
                manual_run[ke] = log[ke].pop(i)
            plt.plot(manual_run["clipped_edges"], manual_run["losses"], 'x', mew=7, markersize=15, color=color, label=None if manual_only else "manual")
     
    if manual_only:
        return

    loss_line = pd.DataFrame({
        "clipped_edges": log["clipped_edges"],
        "losses": log["losses"]
    }).sort_values("clipped_edges")
    loss_line["losses"] = loss_line["losses"].cummin()
        
    if color is not None:
        ax = sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label=f"{k}", marker="o", s=30, color=color)
        ax = sns.lineplot(x=loss_line["clipped_edges"], y=loss_line["losses"], color=color, linewidth=1.5)
    else:
        ax = sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label=f"{k}", marker="o", s=50)
    
    for i,t in enumerate(log['tau']):
        if 'vertices' in log:
            print(t, log["lamb"][i], log['clipped_edges'][i], log['vertices'][i], log['losses'][i])
        else:
            print(t, log["lamb"][i], log['clipped_edges'][i], log['losses'][i])
    return ax

# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

def plot_pareto(pms, log=False, suffix="", order=None, manual=False):
    folder, y_bound, x_bound, task_name = pms

    fig = plt.figure(figsize=(8,8))
    method_list = set()                
    for k, (x, color) in folder.items():
        print(k)

        log_file = f"{x}/post_training{suffix}.pkl"
        if os.path.exists(log_file):
            plot_points(k, log_file, color)
            method_list.add(k)

        if manual:
            manual_log_file = f"{x.rsplit('/', 1)[0]}/acdc/post_training.pkl"
            if os.path.exists(manual_log_file):
                plot_points(k, manual_log_file, color, manual_only=True)           

        if os.path.exists(f"{x}/pre_training.pkl"):
            with open(f"{x}/pre_training.pkl", "rb") as f:
                log = pickle.load(f)
            print(log)
            sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label="pre training", marker="X", s=50)

    plt.xlim(0,x_bound)
    plt.minorticks_on()
    plt.tick_params(which='minor', bottom=False, left=False)

    plt.grid(visible=True, which='major', color='grey', linewidth=0.5)
    plt.grid(visible=True, which='minor', color='darkgoldenrod', linewidth=0.3)
    plt.xlabel(r"Circuit edge count $|\tilde{E}|$")
    plt.ylabel(r"Ablation loss gap $\Delta$")

    def myLogFormat(y,pos):
        # print(y)
        # Find the number of decimal places required
        # decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
        decimalplaces = math.floor(np.log10(y))   # =0 for numbers >=1

        first_digit = str(round(y * 1000)).strip("0.")
        if len(first_digit) == 0:
            return
        if first_digit[0] != "1" and first_digit[0] != "5" and first_digit[0] != "2":
            return ""
        
        if decimalplaces >= 0:
            return first_digit[0] + "".join(decimalplaces * ["0"])
        else:
            return "0." +  "".join((-1- decimalplaces) * ["0"]) + first_digit[0]

    if log:
        plt.yscale("log")
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
        plt.gca().yaxis.set_minor_formatter(ticker.FuncFormatter(myLogFormat))
    else:
        plt.ylim(0,y_bound)

    s = task_name.split("/")
    t = s[0]
    a = s[-1]
    if a in ablation_lookup:
        abl_type = f"{ablation_lookup[a]} ablation"
    else:
        abl_type = f"ablation comparison"
    if order:
        handles, labels = plt.gca().get_legend_handles_labels()
        if "ACDC" in method_list:
            if labels[2] == "manual":
                h2 = plt.Line2D([0], [0], marker='x', markersize=8, mew=4, color=handles[2].get_color(), linestyle='None')
                handles[2] = h2
        if len(handles) == len(order):
            legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right')
        else:
            plt.legend(loc="upper right")
    else:
        plt.legend(loc="upper right")

    plt.suptitle(f"{task_lookup[t]} circuits, {abl_type}")
    plt.tight_layout()

    plt.savefig(f"{plot_folder}/{task_name}_pt_{'log' if log else 'c'}{suffix}.png")
    plt.show()

# %%
l = [
    ("ioi", "cf", 0.2, 1800),
    ("ioi", "oa", 0.14, 1800),
    ("ioi", "mean", 1, 1800),
    ("ioi", "resample", 5, 1800),
    ("gt", "cf", 0.2, 800),
    ("gt", "oa", 0.04, 800),
    ("gt", "mean", 0.4, 800),
    ("gt", "resample", 0.2, 800),
]
for dataset, ablation_type, x_bound, y_bound in l:
    root_folder = f"results/pruning/{dataset}/{ablation_type}"
    ax = None
    folders=({
            "UGS (ours)": (f"{root_folder}/unif", "black"), 
            "HCGS": (f"{root_folder}/hc", "blue"), 
            "ACDC": (f"{root_folder}/acdc", "crimson"),
            # "EP": (f"{root_folder}/ep", "purple"),
            "EAP": (f"{root_folder}/eap", "green")
        }, x_bound, y_bound, f"{dataset}/{ablation_type}")
    for log in [True]:
        plot_pareto(folders, log=log, order=[0,1,4,3,2])

# %%
# ablation comparison results
l2 = [
    ("ioi", 1, 1800),
    ("gt", 0.4, 800)
]
# comparison across ablation types
for dataset, x_bound, y_bound in l2:
    root_folder = f"results/pruning/{dataset}"
    ax = None
    # reg_lambs = [2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
    folders=({
            "Mean": (f"{root_folder}/mean/unif", "indigo"), 
            "Resample": (f"{root_folder}/resample/unif", "olive"), 
            "Optimal": (f"{root_folder}/oa/unif", "black"), 
            "CF": (f"{root_folder}/cf/unif", "maroon"),
        }, x_bound, y_bound, f"{dataset}/comp/unif_")
    for log in [True]:
        plot_pareto(folders, log=log, order=[2,0,1,3], manual=True)

# %%

l3 = [
    ("ioi", "cf", 0.2, 1800),
    ("gt", "resample", 0.2, 800),
]
for dataset, ablation_type, x_bound, y_bound in l3:
    root_folder = f"results/pruning/{dataset}/{ablation_type}"
    ax = None
    # reg_lambs = [2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
    folders=({
            "UGS (ours)": (f"{root_folder}/unif", "black"), 
            "HCGS": (f"{root_folder}/hc_clip", "blue"), 
            # "ACDC": (f"{root_folder}/acdc", "crimson"),
            "EP": (f"{root_folder}/ep", "purple"),
            # "EAP": (f"{root_folder}/eap", "green")
        }, x_bound, y_bound, f"{dataset}/ep-demo/{ablation_type}")
    for log in [True]:
        plot_pareto(folders, log=log, order=[0,1,4,3,2])