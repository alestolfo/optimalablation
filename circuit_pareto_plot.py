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

with open("results/pruning/gt/oa/hc/post_training.pkl", "rb") as f:
    x = pickle.load(f)

for i, l in enumerate(x["lamb"]):
    if l == "0.0002":
        for k in x:
            x[k].pop(i)

with open("results/pruning/gt/oa/hc/post_training.pkl", "wb") as f:
    pickle.dump(x, f)


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

def plot_points(k, x, color=None, suffix=""):
    print(x)
    log_file = f"{x}/post_training{suffix}.pkl"
    print(log_file)
    if os.path.exists(log_file):
        with open(log_file, "rb") as f:
            log = pickle.load(f)
        # print(log)

        for i, lamb in enumerate(log['lamb']):
            if lamb == "manual":
                manual_run = {}
                for ke in log:
                    manual_run[ke] = log[ke].pop(i)
                plt.plot(manual_run["clipped_edges"], manual_run["losses"], 'x', mew=7, markersize=15, color="red", label="manual")

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
    else:
        print("NO POST TRAINING FOUND")
        return
    return ax


# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

def plot_pareto(pms, log=False, suffix="", order=None):
    folder, manual_folder, y_bound, x_bound, task_name = pms

    fig = plt.figure(figsize=(8,8))
    for k, (x, color) in folder.items():
        ax = None
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
        if suffix:
            plot_points(k, x, color, suffix)
        else:
            plot_points(k, x, color)

    for k, (x, color) in manual_folder.items():
        print(k)
        plot_points(k, x, color)
        if os.path.exists(f"{x}/pre_training.pkl"):
            with open(f"{x}/pre_training.pkl", "rb") as f:
                log = pickle.load(f)
            print(log)
            sns.scatterplot(x=log["clipped_edges"], y=log["losses"], label="pre training", marker="X", s=50)

    plt.xlim(0,x_bound)
    # plt.gca().xaxis.set_major_locator(MultipleLocator(200)) # x gridlines every 0.5 units
    # plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2)) # x gridlines every 0.5 units
    plt.minorticks_on()
    plt.tick_params(which='minor', bottom=False, left=False)
    # formatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))

    plt.grid(visible=True, which='major', color='grey', linewidth=0.5)
    plt.grid(visible=True, which='minor', color='darkgoldenrod', linewidth=0.3)
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.01)) # y gridlines every 0.5 units
    plt.xlabel("Edges in circuit")
    plt.ylabel("KL loss")

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
            # print("0." +  "".join((-1- decimalplaces) * ["0"]) + first_digit[0])
            return "0." +  "".join((-1- decimalplaces) * ["0"]) + first_digit[0]
        
    def majorF(y,pos):
        # Find the number of decimal places required
        decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
        # Insert that number into a format string
        formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
        # Return the formatted tick label
        print(formatstring.format(y))
        return formatstring.format(y)


        # return first_digit[0] *
        # print('{{:.{:2d}f}}')
        # # Insert that number into a format string
        # formatstring = '{{:.{:' + first_digit + 'd}f}}'.format(decimalplaces)
        # # Return the formatted tick label
        # print(formatstring.format(y))
        # return formatstring.format(y).replace("1", first_digit)

    if log:
        plt.yscale("log")
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
        plt.gca().yaxis.set_minor_formatter(ticker.FuncFormatter(myLogFormat))
    else:
        plt.ylim(0,y_bound)

    t, a = task_name.split("/", 1)
    if a in ablation_lookup:
        abl_type = f"{ablation_lookup[a]} ablation"
    else:
        abl_type = f"ablation comparison"
    if order:
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels[2] == "manual":
            h2 = plt.Line2D([0], [0], marker='x', markersize=8, mew=4, color='red', linestyle='None')
            handles[2] = h2
        legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    plt.suptitle(f"{task_lookup[t]} circuits, {abl_type}")
    plt.tight_layout()

    plt.savefig(f"{plot_folder}/{task_name}_pt_{'log' if log else 'c'}{suffix}.png")
    plt.show()

# %%
l = [
    ("ioi", "cf", 0.2, 1200),
    ("ioi", "oa", 0.14, 1200),
    ("ioi", "mean", 1, 1200),
    ("ioi", "resample", 5, 1200),
    ("gt", "cf", 0.2, 800),
    ("gt", "oa", 0.04, 800), # need to deal with this
    ("gt", "mean", 0.4, 800),
    ("gt", "resample", 0.2, 800)
]
for dataset, ablation_type, x_bound, y_bound in l:
    root_folder = f"results/pruning/{dataset}/{ablation_type}"
    ax = None
    # reg_lambs = [2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
    folders=({
            # "vertex": "results/pruning_vertices_auto/ioi", 
            "UGS (ours)": (f"{root_folder}/unif", "black"), 
            "HCGS": (f"{root_folder}/hc", "blue"), 
            # "edges uniform window": "results/pruning/ioi/cf/unif_window", 
        }, {
            "ACDC": (f"{root_folder}/acdc", "crimson"),
            "EAP": (f"{root_folder}/eap", "green")
        }, x_bound, y_bound, f"{dataset}/{ablation_type}")
    for log in [True]:
        plot_pareto(folders, log=log, order=[0,1,4,3,2])

# %%
# ablation comparison results
l2 = [
    ("ioi", 1, 1200),
    ("gt", 0.4, 800)
]
# comparison across ablation types
for dataset, x_bound, y_bound in l2:
    root_folder = f"results/pruning/{dataset}"
    ax = None
    # reg_lambs = [2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
    folders=({
            "Mean": (f"{root_folder}/mean/unif", "purple"), 
            "Resample": (f"{root_folder}/resample/unif", "green"), 
            "Optimal": (f"{root_folder}/oa/unif", "black"), 
            "Counterfactual": (f"{root_folder}/cf/unif", "maroon"), 
        }, {}, x_bound, y_bound, f"{dataset}/comp/unif_")
    for log in [True]:
        plot_pareto(folders, log=log, order=[2,0,1,3])

# %%
l3 = [
    ("ioi", "cf", 0.2, 1200),
    ("ioi", "oa", 0.14, 1200),
    ("ioi", "mean", 1, 1200),
    ("ioi", "resample", 5, 1200),
    ("gt", "cf", 0.2, 800),
    ("gt", "oa", 0.04, 800), # need to deal with this
    ("gt", "mean", 0.4, 800),
    ("gt", "resample", 0.2, 800)
]
for dataset, ablation_type, x_bound, y_bound in l:
    root_folder = f"results/pruning/{dataset}"
    ax = None
    # reg_lambs = [2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
    other_folders = {
            "Mean ablation": (f"{root_folder}/mean/acdc", "goldenrod"), 
            "Resample ablation": (f"{root_folder}/resample/acdc", "royalblue"), 
            "Optimal ablation": (f"{root_folder}/oa/acdc", "black"), 
            "Counterfactual": (f"{root_folder}/cf/acdc", "purple"), 
        }
    my_type = ablation_lookup[ablation_type].capitalize()
    print(my_type)
    my_folder = other_folders[my_type]
    del other_folders[my_type]
    folders=(other_folders, {
            my_type: my_folder,
            # "ACDC": (f"{root_folder}/acdc", "black"),
            # "EAP": (f"{root_folder}/eap", "green")
        }, x_bound, y_bound, f"{dataset}/{ablation_type}")
    for log in [False, True]:
        plot_pareto(folders, log=log, suffix=f"_{ablation_type}")

# %%
folders=[
    ({
        # "vertex": "results/pruning_vertices_auto/ioi", 
        "edges HC": "results/pruning_edges_auto/hc", 
        # "edges HC (vertex prior)": "results/pruning/ioi/oa/vertex_prior", 
        "edges uniform": "results/pruning/ioi/oa/unif", 
        # "edges uniform window": "results/pruning/ioi/oa/unif_window", 
    }, {
        "ACDC": "results/pruning/ioi/oa/acdc",
        # "eap": "results/pruning/ioi/oa/eap"
    }, 0.15, 1500, "ioi"),
    ({
        # "vertex": "results/pruning_vertices_auto/gt", 
        # "edges HC": "results/pruning/gt/oa/edges", 
        # "edges HC (vertex prior)": "results/pruning/gt/oa/vertex_prior", 
        "edges uniform": "results/pruning/gt/oa/unif", 
    }, {
        "ACDC": "results/pruning/gt/oa/acdc",
        "eap": "results/pruning/gt/oa/eap"
    }, 0.05,1000,"gt"),
]


for folder in folders:
    plot_pareto(folder)

# %%


def compare_train_curves(folder_1, folder_2, edge_assn=False):
    edge_lookup_1 = {}
    edge_lookup_2 = {}
    if os.path.exists(f"{folder_1}/post_training.pkl"):
        with open(f"{folder_1}/post_training.pkl", "rb") as f:
            log = pickle.load(f)
        # print(log)
        for i, edges in enumerate(log['edges']):
            edge_lookup_1[log['lamb'][i]] = edges

    if os.path.exists(f"{folder_2}/post_training.pkl"):
        with open(f"{folder_2}/post_training.pkl", "rb") as f:
            log = pickle.load(f)
        # print(log)
        for i, edges in enumerate(log['edges']):
            edge_lookup_2[log['lamb'][i]] = edges
    
    print(edge_lookup_1)
    print(edge_lookup_2)
    edge_corr = {}
    if edge_assn:
        for lamb in edge_lookup_1:
            cur_edge_diff = 10000
            for lamb_2 in edge_lookup_2:
                edge_diff = abs(edge_lookup_2[lamb_2] - edge_lookup_1[lamb])
                if edge_diff < cur_edge_diff:
                    cur_edge_diff = edge_diff
                    edge_corr[lamb] = lamb_2

    for path in glob.glob(f"{folder_1}/*"):
        lamb = path.split("/")[-1]

        if not os.path.exists(f"{folder_1}/{lamb}/"):
            continue

        if edge_assn:
            lamb_2 = edge_corr[lamb]
        else:
            lamb_2 = lamb

        if os.path.exists(f"{folder_1}/{lamb}/fit_loss_log.pkl") and os.path.exists(f"{folder_2}/{lamb_2}/fit_loss_log.pkl"):
            with open(f"{folder_1}/{lamb}/fit_loss_log.pkl", "rb") as f:
                train_curve_1 = pickle.load(f)
            with open(f"{folder_2}/{lamb_2}/fit_loss_log.pkl", "rb") as f:
                train_curve_2 = pickle.load(f)
            
            if lamb in edge_lookup_1:
                print("edges control:", edge_lookup_1[lamb])
            if lamb_2 in edge_lookup_2:
                print("edges new:", edge_lookup_2[lamb_2])
            
            train_curve_1.compare_plot("kl_loss", 50, train_curve_2, f"Post training comparison {lamb}", start=500)
        
        if os.path.exists(f"{folder_1}/{lamb}/metadata.pkl") and os.path.exists(f"{folder_2}/{lamb_2}/metadata.pkl"):
            with open(f"{folder_1}/{lamb}/metadata.pkl", "rb") as f:
                train_curve_1 = pickle.load(f)[0]
            with open(f"{folder_2}/{lamb_2}/metadata.pkl", "rb") as f:
                train_curve_2 = pickle.load(f)[0]
            train_curve_1.compare_plot("kl_loss", 50, train_curve_2, f"Training comparison {lamb}", start=300)

            train_curve_1.compare_plot("complexity_loss", 50, train_curve_2, f"Training comparison {lamb}", start=300)
# %%
# comparing ioi with diverse dataset to templated dataset
compare_train_curves("results/pruning/ioi/oa/b_unif_wrong_4", "results/pruning-5-6/ioi_edges_unif")

# %%
compare_train_curves("results/pruning/ioi/oa/dynamic_unif", "results/pruning/ioi/oa/unif_correct")

# %%

compare_train_curves("results/pruning/ioi/oa/dynamic_unif-0.5", "results/pruning/ioi/oa/dynamic_unif-0.99")



# %%
# unif: ZERO node reg, detached bottom derivative
# wrong: attached bottom derivative with zero node reg
# wrong_2: attached bottom derivative, fixed node_reg to 5e-3
# wrong_3: attached bottom derivative, fixed node_reg to 5e-4
# correct: scaling node_reg, detached bottom derivative

# wrong_3 ioi_b: corrected diverse dataset predicting first token of IO
# wrong_4 ioi_b: wrong diverse dataset, sometimes predicting IO completion

# ioi scale invariance: fix overweighting of tail probs (dividing by smaller window)
# scale_var: fix overweighting of bottom
compare_train_curves("results/pruning-5-6/gt_edges_unif", "results/pruning/gt/oa/unif_wrong")

# %%
# 
compare_train_curves("results/pruning/ioi/oa/acdc", "results/pruning/ioi/oa/unif", edge_assn=True)