# %%
import torch
import json
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from glob import glob
from functools import partial
import os
from sys import argv
import torch.optim
import time
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from itertools import cycle
from utils.training_utils import load_model_data, LinePlot, gen_resample_perm
from torch.utils.data import DataLoader
from utils.tracing_utils import get_subject_tokens, ct_inference
from matplotlib.scale import FuncScale
from matplotlib.ticker import FuncFormatter
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as mtransforms


# %%

sns.set()

model_name = "gpt2-xl"
mode="fact"
ds_name = "my_facts" if mode == "fact" else "my_attributes"
ds_file = "combined" if mode == "fact" else None
ds_path = "utils/datasets/facts"
base_folder=f"results/causal_tracing/{mode}"
plot_folder=f"plots_export/causal_tracing"

if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# %%
# test_start = math.ceil(0.6 * len(correct_prompts))
test_start = 0
node_types = {"attn": "Attention", "mlp": "MLP"}
labels={"gauss": "Gaussian noise", "oa": "Optimal ablation"}
colors={"oa": "black", "gauss": "red"}
token_types={"last": "last token", "last_subject": "last subject token", "all_subject": "all subject tokens"}

clean_means = []
denominator_vars = {l: [] for l in labels}
corrupted_means = {l: [] for l in labels}
settings_list = [
    (w,x) 
    for w in [
        0, 2, 4
    ] for x in token_types
]
graph_list = [
    (y,z)
    for y in node_types
    for z in labels
]
clean_mean = 0

# compute clean and corrupted means
for window_size, token_type in settings_list:
    for node_type, ablate_type in graph_list:
        folder = f"{base_folder}/{token_type}/{node_type}/{window_size}"

        clean_probs = torch.load(f"{folder}/{ablate_type}_clean_probs.pth")
        corrupted_probs = torch.load(f"{folder}/{ablate_type}_corrupted_probs.pth")

        clean_mean = clean_probs.mean().item()
        corrupted_mean = corrupted_probs.mean().item()

        denominator_vars[ablate_type].append((clean_probs.mean(dim=1) - corrupted_probs.mean(dim=1)).var().item())

        clean_means.append(clean_mean)
        corrupted_means[ablate_type].append(corrupted_mean)

# %%
agg_clean_mean = np.mean(clean_means)

for l in labels:
    # take the best constant?
    corrupted_means[l] = np.mean(corrupted_means[l])
    denominator_vars[l] = np.mean(denominator_vars[l])

# %%

CORR_SIZE = 18
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=CORR_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%

all_probs = []
desc = []

# %%
df = []
for window_size, token_type in settings_list:
    f, axes = plt.subplots(1,2, figsize=(16,4))
    for i,node_type in enumerate(node_types):
        folder = f"{base_folder}/{token_type}/{node_type}/{window_size}"

        for ablate_type in labels:
            aie = torch.load(f"{folder}/{ablate_type}_aie.pth")
            clean_probs = torch.load(f"{folder}/{ablate_type}_clean_probs.pth")
            spec_baseline = torch.load(f"{folder}/{ablate_type}_corrupted_probs.pth")
            
            aie = aie.mean(dim=1).flatten(start_dim=0, end_dim=1)
            clean_probs = clean_probs.mean(dim=1).flatten(start_dim=0, end_dim=1)
            spec_baseline = spec_baseline.mean(dim=1).flatten(start_dim=0, end_dim=1)

            numerator = (clean_probs[:,None] - aie).clamp(min=0)
            denominator = clean_probs - spec_baseline

            est_numerator = numerator.mean(dim=0)
            low_var_denominator = agg_clean_mean - corrupted_means[ablate_type]

            asymptotic_variances = []
            for layer in range(numerator.shape[-1]):
                covariance = torch.stack([numerator[:, layer], denominator], dim=0).cov()

                # estimator for asymptotic variance of ratios
                asy_var = (
                    (est_numerator[layer] / low_var_denominator).square().item()
                    * (
                        (covariance[0,0] / est_numerator[layer].square())
                        + (denominator_vars[ablate_type] / 
                        (low_var_denominator * low_var_denominator))
                        - 2 * (covariance[0,1] / (est_numerator[layer] * low_var_denominator))
                    )
                )
                asymptotic_variances.append(asy_var.item())
            
            asymptotic_variances = np.array(asymptotic_variances) / numerator.shape[0]

            negatives = asymptotic_variances[asymptotic_variances < 0]
            if len(negatives) > 0:
                print(negatives)
            ci = 1.96 * np.sqrt(asymptotic_variances.clip(min=0))

            aie_means = (1 - est_numerator / low_var_denominator).cpu().numpy()
            aie_ci_lb = (aie_means - ci).clip(min=0)
            aie_ci_ub = (aie_means + ci).clip(min=0)

            aie_means = aie_means.clip(min=0)

            # aie = (aie - corrupted_means[ablate_type]) / (agg_clean_mean - corrupted_means[ablate_type])
            # aie_means = aie.mean(dim=[0,1,2]).cpu().numpy()

            all_probs.append(aie_means)
            desc.append((token_type, window_size, node_type, ablate_type))

            plt.sca(axes[i])
            sns.scatterplot(
                x=np.arange(aie_means.shape[0]) + (0.15 if ablate_type=="oa" else -0.15), 
                y=aie_means, 
                label=labels[ablate_type], 
                color=colors[ablate_type])
            
            bar_cont = plt.bar(
                np.arange(aie_means.shape[0]) + (0.15 if ablate_type=="oa" else -0.15), 
                aie_ci_ub - aie_ci_lb,
                0.1,
                bottom=aie_ci_lb,
                color=colors[ablate_type],
                linewidth=0
            )

            df.append({
                "window": window_size,
                "token": token_type,
                "node": node_type,
                "ablate": ablate_type,
                **{k: aie_means[k] for k in range(aie_means.shape[0])}
            })

            # bar_cont = plt.bar(
            #     np.arange(aie_means.shape[0]) + (0.15 if ablate_type=="oa" else -0.15), 
            #     aie_means,
            #     0.3, 
            #     label=labels[ablate_type], 
            #     color=colors[ablate_type]
            # )

            # sample_size = clean_probs.shape[0] * clean_probs.shape[-1]
            # bounds = agg_clean_mean / (agg_clean_mean - baseline)
            # hoeffding = bounds * math.sqrt(math.log(2/0.05) / (2 * sample_size))
            # print(hoeffding)
            # line = plt.axhline(y=hoeffding,color=colors[ablate_type], linestyle=":")

        # axes[i].set_yscale('function', functions=(lambda x: np.copysign(np.abs(x) ** (3. / 5), x), lambda x: np.copysign(np.abs(x) ** (5. / 3), x)))
        # axes[i].set_ylim(-0.2, 1)

        axes[i].set(xlabel="Layer number", ylabel="AIE")
        axes[i].set_title(f"AIE, {node_types[node_type]} layers")
        axes[i].legend()

    y1_min, y1_max = axes[0].get_ylim()
    y2_min, y2_max = axes[1].get_ylim()
    common_min = min(y1_min, y2_min) - 0.01
    common_max = max(y1_max, y2_max)
    axes[0].set_ylim(common_min, common_max)
    axes[1].set_ylim(common_min, common_max)

    plt.suptitle(f"AIE (proportion probability recovered) patching at {token_types[token_type]}, window size {(window_size * 2 + 1)}")

    plt.tight_layout()
    plt.savefig(f"{plot_folder}/{token_type}_{window_size}.png")
    plt.show()

# %%
print(agg_clean_mean)
print(corrupted_means)
pd.set_option("display.max_columns", 100)
pd.DataFrame(df)
# %%



# sns.scatterplot(x=corrupted_probs.flatten().cpu(), y=clean_probs.flatten().cpu(), s=5)
# # %%
# sns.scatterplot(x=corrupted_probs.flatten().cpu(), y=aie.mean(dim=-1).flatten().cpu(), s=5)
# sns.lineplot(x=[0,1],y=[0,1])

# # %%
# sns.scatterplot(x=clean_probs.flatten().cpu(), y=aie.mean(dim=-1).flatten().cpu(), s=5)
# sns.lineplot(x=[0,1],y=[0,1])

# # %%

# sns.lineplot((aie - corrupted_probs.unsqueeze(-1)).std(dim=[0,1,2]).cpu(), label="corr_std")
# sns.lineplot((aie - clean_probs.unsqueeze(-1)).std(dim=[0,1,2]).cpu(), label="clean_std")
# sns.lineplot(aie.std(dim=[0,1,2]).cpu(), label="clean_std")

# %%
    # # lcb = sns.lineplot(aie_means - 1.96 * aie_stds / math.sqrt(n_samples), color=ax[0].get_facecolor())
    # # ucb = sns.lineplot(aie_means + 1.96 * aie_stds / math.sqrt(n_samples), color=ax[0].get_facecolor())
    # ax = sns.lineplot(x=[0, len(aie_means)], y=[0, 0], color="black")
    # # for i, t in enumerate(ax.get_xticklabels()):
    # #     if (i % 5) != 0:
    # #         t.set_visible(False)



# target probs: [batch]. [10,]
# corrupted probs: [batch, layers]
    #         aie = torch.load(f"{folder}/aie.pth")
    #         corrupted_probs = torch.load(f"{folder}/{s1}/{s3}_{s2}_corrupted_probs.pth")

    #         corrupted_probs = corrupted_probs[test_start:]

    #         aie_means = []
    #         aie_stds = []
    #         n_samples = aie[0].nelement()
    #         for i in range(len(aie)):
    #             aie[i] = aie[i][test_start:]
    #             aie_means.append(aie[i].mean().item() - corrupted_probs.mean().item())
    #             aie_stds.append(aie[i].std().item())
    #         print(labels[s2])
    #         intv = np.arange(len(aie_means))
    #         aie_means = np.array(aie_means)
    #         aie_stds = np.array(aie_stds)

    #         width = 0.5
    # break
        

# %%
sns.histplot(all_corrupted_means['oa'])
# %%
