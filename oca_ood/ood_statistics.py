# %%

import torch
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# sns.set()

# %%

folder = "oca/ioi"
baseline_folder = "oca/ioi"
out_folder = "oca_ood/ioi-ioi"
# %%
with open(f"{baseline_folder}/baseline_resid.pkl", "rb") as f:
    baseline_dist = pickle.load(f)

comp_classes = ["zero", "resample", "mean", "mode"]
comp_sets = {}

for k in comp_classes:
    with open(f"{folder}/{k}_ablation_resid.pkl", "rb") as f:
        comp_sets[k] = pickle.load(f)

# %%
if baseline_folder != folder:
    with open(f"{folder}/baseline_resid.pkl", "rb") as f:
        comp_sets['subtask_baseline'] = pickle.load(f)

# %%
# baseline calculations
stat_types = ['kde','nn','pca','qt']
# benchmarks = {'kde': [0.4, 0.5, 0.6, 0.7, 0.8], 'nn': [10,20,50,100,500]}
benchmarks = {'kde': [0.6, 0.7, 0.8], 'nn': [50,100,500]}

# %%
dist_taus = []

if not os.path.exists(f"{baseline_folder}/baseline_nn.pkl"):
    for layer_no in range(len(baseline_dist)):
        dist_tau = []
        for idx in tqdm(range(0,baseline_dist[layer_no].shape[0],10)):
            dist = -(baseline_dist[layer_no][idx:idx+10].unsqueeze(1)-baseline_dist[layer_no]).norm(dim=-1)
            vals, idxs = dist.topk(501, dim=1)
            dist_tau.append(vals[:,benchmarks['nn']])
        dist_tau = torch.cat(dist_tau, dim=0)
        dist_taus.append(dist_tau)

    with open(f"{baseline_folder}/baseline_nn.pkl", "wb") as f:
        pickle.dump(dist_taus, f)
# %%
with open(f"{baseline_folder}/baseline_nn.pkl", "rb") as f:
    dist_taus = pickle.load(f)
# %%
    
stds = []
baseline_means = []
eigenvalues = []
pcs = []
for layer_no in range(len(baseline_dist)):
    stds.append(baseline_dist[layer_no].var(dim=0).sum().sqrt())
    baseline_means.append(baseline_dist[layer_no].mean(dim=0))
    u, s, v = torch.pca_lowrank(baseline_dist[layer_no], q=240)
    eigenvalues.append(s)
    pcs.append(v)

# %%
    
def compute_ood(point, layer_no, std):
    all_distances = (point - baseline_dist[layer_no]).norm(dim=-1)

    kde = []
    # KDE
    for dist in benchmarks['kde']:
        kde_est = (all_distances < dist * std).sum()
        kde.append(kde_est)

    # NN
    nn = []
    dists, idxs = (-all_distances).topk(benchmarks['nn'][-1]+1)
    for i,t in enumerate(benchmarks['nn']):
        nn_est = (dists[:t+1] > dist_taus[layer_no][idxs[:t+1],i]).sum()
        nn.append(nn_est)
    
    # PCA
    pca_stat = (torch.matmul(point - baseline_means[layer_no], pcs[layer_no][:,:200]) / s[:200]).abs().sum()

    # marginal rank tests
    quantiles = ((point < baseline_dist[layer_no]) * 1.).mean(dim=0)
    qt_stat = (quantiles - 0.5).square().sum()
    
    return {'kde': kde, 'nn': nn, 'pca': pca_stat.item(), 'qt': qt_stat.item()}

# %%
n_layers = 12
all_statistics = {stat_type: {i:{} for i in range(n_layers)} for stat_type in stat_types}

def retrieve_stats(dist, dist_name):
    for layer_no in range(1,n_layers):
        order = torch.randperm(dist[layer_no].shape[0])

        for stat_type in stat_types:
            all_statistics[stat_type][layer_no][dist_name] = []

        for i in tqdm(range(500)):
            point = dist[layer_no][order[i]]
            stat_obj = compute_ood(point, layer_no, stds[layer_no])

            for stat_type in stat_obj:
                all_statistics[stat_type][layer_no][dist_name].append(stat_obj[stat_type])
        for stat_type in stat_obj:
            all_statistics[stat_type][layer_no][dist_name] = torch.tensor(all_statistics[stat_type][layer_no][dist_name]).float()

# %%
retrieve_stats(baseline_dist, "baseline")

# %%
for k in comp_sets:
    retrieve_stats(comp_sets[k], k)

# %%

def generate_stat_plot(stats_by_dist, file_name, title, j=None):
    f, axes = plt.subplots(1,2, figsize=(10,5))
    # sns.lineplot(x=(torch.arange(ref_x.shape[0]) / ref_x.shape[0]).cpu(),y=(torch.arange(ref_x.shape[0]) / ref_x.shape[0]).cpu(), ax=axes[0,0])
    for dist_name in stats_by_dist:
        if j is None:
            ref_x = stats_by_dist['baseline'].sort()[0]
            series = stats_by_dist[dist_name]
        else:
            ref_x = stats_by_dist['baseline'][:,j].sort()[0]
            series = stats_by_dist[dist_name][:,j]
        y = (series.unsqueeze(-1) > ref_x).float().mean(dim=-1).clone().detach().sort()[0]
        sns.lineplot(x=(torch.arange(ref_x.shape[0]) / ref_x.shape[0]).cpu(), y=y.cpu(), label=dist_name, ax=axes[0])
        sns.histplot(series.cpu(),label=dist_name, bins=100, ax=axes[1])
        print(dist_name, series.float().mean(), series.float().quantile(.5))
    
    plt.suptitle(title)
    plt.legend()
    plt.savefig(file_name)
    plt.close()
# %%

for stat_type in stat_types:
    for k in all_statistics[stat_type]:
        if len(all_statistics[stat_type][k].keys()) > 0:
            if stat_type in benchmarks:
                for j in range(len(benchmarks[stat_type])):
                    file_name = f"{out_folder}/{stat_type}_layer{k}_{j}.png"
                    generate_stat_plot(all_statistics[stat_type][k], file_name, f"{stat_type} layer {k} threshold {benchmarks[stat_type][j]}", j)
            else:
                file_name = f"{out_folder}/{stat_type}_layer{k}.png"
                generate_stat_plot(all_statistics[stat_type][k], file_name, f"{stat_type} layer {k}")
# %%
