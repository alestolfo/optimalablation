# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
import os
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
from sys import argv
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, LinePlot, plot_no_outliers
from utils.lens_utils import LensExperiment, compile_loss_dfs, corr_plot, overall_comp

import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.set()

model_name = "gpt2-xl"


folders = {
    "modal": f"results/lens/{model_name}/oa",
    "linear_oa": f"results/lens/{model_name}/linear_oa",
    "tuned": f"results/lens/{model_name}/tuned",
}

data_folder="results/lens/truth"
out_folder="plots_export/lens/truth"

for k in folders:
    if not os.path.exists(folders[k]):
        os.makedirs(folders[k])

if model_name == "gpt2-xl":
    CAUSAL_BATCH_SIZE = 3
elif model_name == "gpt2-large":
    CAUSAL_BATCH_SIZE = 7
elif model_name == "gpt2-medium":
    CAUSAL_BATCH_SIZE = 12
elif model_name == "gpt2-small":
    CAUSAL_BATCH_SIZE = 25
else:
    raise Exception("Model not found")

# %%

from utils.datasets.truth.dev.prefixes import *
from utils.datasets.truth.dev.metrics import *
from utils.datasets.truth.data.dataset_params import *
from utils.datasets.truth.data.demo_params import *
from utils.datasets.truth.data.prompt_params import *
from utils.datasets.truth.model.model_params import *

# %%

model_name = 'gpt2-xl'
setting='permuted_incorrect_labels'
demo_params = DEMO_PARAMS[setting]
model_params = MODEL_PARAMS[model_name.replace("-", "_")]
# %%
# model_name = "gpt2-small"
batch_size = CAUSAL_BATCH_SIZE * 30
# 100K OWT samples with default sequence length: 235134
device, model, tokenizer, owt_iter = load_model_data("gpt2-small", batch_size)
model = HookedTransformer.from_pretrained(model_name, device=device)
exp = LensExperiment(model, owt_iter, folders, device)

# %%
label_modes = ['True labels', 'Permuted labels']
lens_names = ['modal', 'tuned']

# %%
num_inputs = 2000
n_demos = 10

for dataset_name in DATASET_PARAMS:
    if not os.path.exists(f"{data_folder}/{dataset_name}.pkl"):
        print(f"Analyzing dataset {dataset_name}")

        dataset_params = DATASET_PARAMS[dataset_name]
        prompt_params = PROMPT_PARAMS[dataset_name]

        prefixes = Prefixes(
            get_dataset(dataset_params),
            prompt_params[0],
            demo_params,
            model_params,
            tokenizer,
            num_inputs,
            n_demos,
        )

        tokenized_inputs = prefixes.true_prefixes_tok + prefixes.false_prefixes_tok
        tok_prec_label_indx = (
            prefixes.true_prefixes_tok_prec_label_indx
            + prefixes.false_prefixes_tok_prec_label_indx
        )
        lab_first_token_ids = prefixes.lab_first_token_ids

        output_probs = {'modal': [], 'tuned': []}

        for i, (t_inp, indices) in enumerate(
            zip(tqdm(tokenized_inputs), tok_prec_label_indx)
        ):
            with torch.no_grad():
                output_probs_, _ = exp.get_lens_loss(
                    t_inp['input_ids'].to(device), return_probs=True, token_positions=indices)
            
            def normalize_probs(probs):
                return (probs + 1e-14) / (
                    probs.sum(dim=-1, keepdim=True) + (1e-14 * probs.shape[-1])
                )

            for lens_name in ['modal', 'tuned']:
                probs = output_probs_[lens_name][...,lab_first_token_ids]
                output_probs[lens_name].append(normalize_probs(probs))

        for lens_name in ['modal', 'tuned']:
            output_probs[lens_name] = rearrange(
                torch.stack(output_probs[lens_name], dim=0),
                "(n_prefix n_inputs) n_demo n_layer lab_space_size -> n_prefix n_inputs n_layer n_demo lab_space_size",
                n_prefix = 2,
                n_inputs = len(output_probs[lens_name]) // 2
            )

        cal_correct_over_incorrect = {}
        for lens_name in ['modal', 'tuned']:
            print("*Get quantile probabilities of labels for calibration.*", flush=True)
            n_labels = output_probs[lens_name].shape[-1]
            quantiles, means = get_thresholds(output_probs[lens_name], n_labels)

            print("*Compute cal_correct_over_incorrect metric.*", flush=True)
            cal_correct_over_incorrect[lens_name] = get_cal_correct_over_incorrect(
                output_probs[lens_name], quantiles, prefixes.true_prefixes_labels
            )

        with open(f"{data_folder}/{dataset_name}.pkl", "wb") as f:
            pickle.dump(cal_correct_over_incorrect, f)

# %%

agg_data = {}
for dataset_name in DATASET_PARAMS:
    agg_data[dataset_name] = {}
    with open(f"{data_folder}/{dataset_name}.pkl", "rb") as f:
        cal_correct_over_incorrect = pickle.load(f)

    for ax_idx, n_demos in enumerate([1,3,5,7,10]):

        agg_data[dataset_name][n_demos] = {}

        for i, label_mode in enumerate(label_modes):
            agg_data[dataset_name][n_demos][label_mode] = {}

            for j, lens_name in enumerate({'modal', 'tuned'}):
                yvals = cal_correct_over_incorrect[lens_name][0][i][...,n_demos-1].flatten()
                agg_data[dataset_name][n_demos][label_mode][lens_name] = yvals
        
with open(f"{data_folder}/agg.pkl", "wb") as f:
    pickle.dump(agg_data, f)


# %%
blue_shades = plt.cm.Greens(np.linspace( 1, 0.5, 2))
red_shades = plt.cm.Reds(np.linspace( 0.8, 0.5, 2))

lineshades = {'True labels': blue_shades, 'Permuted labels': red_shades}

def plot_figure(data, dataset_name, demos, string_labels=False):
    CORR_SIZE = 20
    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 32

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=CORR_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(1,3,figsize=(17,5))
    lens_captions = {'modal': "OCA", 'tuned': "Tuned"}

    for ax_idx, n_demos in enumerate(demos):
        for label_mode in label_modes:
            for j, lens_name in enumerate(lens_names):
                # mean, UCB, LCB
                yvals = data[n_demos][label_mode][lens_name]
                sns.lineplot(
                    x=np.arange(yvals.shape[0]),
                    y=yvals,
                    label=f"{label_mode}, {lens_captions[lens_name]}".replace("labels", "demos").replace("Permuted", "False"),
                    ax=axes[ax_idx],
                    color=lineshades[label_mode][j],
                    linestyle="solid" if lens_name == "tuned" else "dashed",
                    alpha=0.7
                )
            axes[ax_idx].axhline(y=data[n_demos][label_mode]['modal'][-1], color="black" if label_mode == "True labels" else "red", linestyle="dotted", label=f"{label_mode},\n full model".replace("labels", "demos").replace("Permuted", "False"))
        axes[ax_idx].set(xlabel="Layer number")
        axes[ax_idx].set_title(n_demos if string_labels else f"{n_demos} demos")
        axes[ax_idx].get_legend().remove()

    axes[0].set(ylabel="Calibrated accuracy")
    axes[-1].legend(bbox_to_anchor=(1.05, 1.05))

    plt.suptitle(f"Elicitation accuracy on {dataset_name.replace('_', ' ')}, GPT-2-XL")
    plt.tight_layout()

    plt.savefig(f"{out_folder}/{dataset_name}.png")

# %%

demo_plot = [5,7,10]

with open(f"{data_folder}/agg.pkl", "rb") as f:
    agg_data = pickle.load(f)

del agg_data['sst2_ab']

# for dataset_name in agg_data:
#     plot_figure(agg_data[dataset_name], dataset_name, demo_plot)
# %%
combined_stats = {
    n_demos: {
        label_mode: {
            lens_name: np.mean(np.stack([
                agg_data[dataset_name][n_demos][label_mode][lens_name]
                for dataset_name in agg_data
            ], axis=0), axis=0)
            for lens_name in lens_names
        } for label_mode in label_modes
    } for n_demos in demo_plot
}

# plot_figure(combined_stats, "all datasets", demo_plot)

# %%

demo_plot = [10]

CORR_SIZE = 12
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=CORR_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=CORR_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ds_list = list(agg_data.keys())
comparative_stats_plot = {
    n_demos: {
        label_mode: {
            lens_name: np.array([
                np.max(agg_data[dataset_name][n_demos][label_mode][lens_name])
                - agg_data[dataset_name][n_demos][label_mode]['modal'][-1]
                for dataset_name in ds_list
            ])
            for lens_name in lens_names
        } for label_mode in label_modes
    } for n_demos in demo_plot
}

for n_demos in demo_plot:
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    for i, label_mode in enumerate(label_modes):
        plt.sca(axes[i])
        plt.title(f"{label_mode}")
        sns.scatterplot(
            x=comparative_stats_plot[n_demos][label_mode]['tuned'], 
            y=comparative_stats_plot[n_demos][label_mode]['modal'],
            # label=label_mode,
            color=lineshades[label_mode][0],
        )

        data_series = list(zip(
            ds_list,
            comparative_stats_plot[n_demos][label_mode]['tuned'], 
            comparative_stats_plot[n_demos][label_mode]['modal']
        ))
        restricted_series = [
            (ds, xval, yval) for ds, xval, yval in data_series 
            if xval > yval + 0.01 or xval < yval - 0.028
        ]

        for ds, xval, yval in data_series:
            if xval > yval + 0.01 or xval < yval - 0.028:
                va = "bottom" if any([
                        xopp - xval < 0.02
                        and xopp - xval > -0.01
                        and yval - yopp < 0.02
                        and yval - yopp > -0.01
                        for _, xopp, yopp in data_series
                    ]) else "top"
                ha = "right" if any([
                        xopp - xval < 0.03
                        and xopp - xval > 0
                        and yval - yopp < 0.02
                        and yval - yopp > -0.01
                        for _, xopp, yopp in restricted_series
                    ]) else "left"
                # print(ds, ha)
                plt.annotate(
                    ds, 
                    (xval, yval),
                    ha=ha,
                    va=va
                )
    
        common_min = min(plt.gca().get_xlim()[0], plt.gca().get_ylim()[0])
        common_max = max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])

        sns.lineplot(x=[common_min, common_max], y=[common_min, common_max], linestyle="dotted", color="black")
        plt.gca().set(
            xlabel="Tuned lens",
            ylabel="OCA lens"
        )

    plt.suptitle(f"Elicitation accuracy boost comparison")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/scatter.png")
    plt.show()

# %%
n_demos = 10

main_figure = {
    "Average of 15 datasets": combined_stats[n_demos],
    "DBPedia": agg_data["dbpedia"][n_demos],
    "MRPC": agg_data["mrpc"][n_demos]
}
plot_figure(main_figure, f"selected datasets with {n_demos} demos", ["Average of 15 datasets", "DBPedia", "MRPC"], string_labels=True)

# %%

n_demos = 10

df = [
    {"demos": n_demos, "labels": label_mode, "lens": lens_name, "dataset": ds, "acc": comparative_stats_plot[n_demos][label_mode][lens_name][i]}
    for n_demos in comparative_stats_plot
    for label_mode in comparative_stats_plot[n_demos]
    for lens_name in comparative_stats_plot[n_demos][label_mode]
    for i, ds in enumerate(ds_list)
]

df = pd.DataFrame(df)

tuned_df = df[df["lens"] == "tuned"].rename(columns={"acc": "tuned"}).drop(columns=["lens"])
oca_df = df[df["lens"] == "modal"].rename(columns={"acc": "oca"}).drop(columns=["lens"])
comp_df = tuned_df.merge(oca_df, on=["demos", "labels", "dataset"])
comp_df["diff"] = comp_df.eval("oca - tuned")
# %%

n_demos = 10

CORR_SIZE = 28
SMALL_SIZE = 18
MEDIUM_SIZE = 24
BIG_SIZE = 28
BIGGER_SIZE = 48

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=CORR_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axes = plt.subplots(4,4,figsize=(20,20))
lens_captions = {'modal': "OCA", 'tuned': "Tuned"}

for ax_idx, dataset_name in enumerate(agg_data):
    plt.sca(axes[ax_idx // 4, ax_idx % 4])

    for label_mode in label_modes:
        for j, lens_name in enumerate(lens_names):
            # mean, UCB, LCB
            yvals = agg_data[dataset_name][n_demos][label_mode][lens_name]
            sns.lineplot(
                x=np.arange(yvals.shape[0]),
                y=yvals,
                label=f"{label_mode}\n {lens_captions[lens_name]} lens".replace("labels", "demos").replace("Permuted", "False"),
                color=lineshades[label_mode][j],
                linestyle="solid" if lens_name == "tuned" else "dashed",
                alpha=0.7
            )
        baseline = agg_data[dataset_name][n_demos][label_mode]["modal"][-1]
        plt.gca().axhline(y=baseline, color="black" if label_mode == "True labels" else "red", linestyle="dotted")

    # plt.gca().set(ylabel="Accuracy", xlabel="Layer number")
    plt.gca().set_title(dataset_name.replace("_", " ").replace(" pairs", "").replace(" stance", ""))
    plt.gca().get_legend().remove()

    if ax_idx % 4 == 0:
        plt.gca().set(ylabel="Calibrated accuracy")
    
    if ax_idx // 4 == 3:
        plt.gca().set(xlabel="Layer number")

axes[-1,-1].set_axis_off()

plt.suptitle(f"Elicitation accuracy on text classification datasets, GPT-2-XL")
plt.tight_layout()
plt.gca().legend(bbox_to_anchor=(1.05, 1.1))

plt.savefig(f"{out_folder}/all.png")

# %%
