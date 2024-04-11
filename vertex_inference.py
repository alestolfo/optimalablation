# %%
import torch
from sys import argv
import numpy as np 
from tqdm import tqdm
import torch.optim
import glob
import seaborn as sns
import pickle
from VertexPruner import VertexPruner
from MaskSampler import ConstantMaskSampler
from MaskConfig import VertexInferenceConfig
from task_datasets import IOIConfig, GTConfig
from circuit_utils import discretize_mask, prune_dangling_edges, retrieve_mask, mask_to_nodes, nodes_to_mask
from training_utils import load_model_data, LinePlot

# %%
# load model
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
# model.cfg.use_attn_result = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
# settings
try:
    reg_lamb = float(argv[1])
except:
    reg_lamb=1e-2

base_folder = f"pruning_vertices_auto/ioi_with_mlp"

batch_size = 50
pruning_cfg = VertexInferenceConfig(model.cfg, device, None, batch_size=batch_size)
pruning_cfg.lamb = reg_lamb
pruning_cfg.n_samples = 1

task_ds = IOIConfig(batch_size, device)
ds_test = task_ds.get_test_set(tokenizer)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
vertex_pruner.add_patching_hooks()

# %%

for g in glob.glob(f"{base_folder}/*"):
    reg_lamb = g.split("/")[-1]
    try:
        float(reg_lamb)
    except:
        continue
    print(reg_lamb)
    folder=f"{base_folder}/{reg_lamb}"
    out_path=f"{base_folder}/report/{str(reg_lamb).replace('.', '-')}.pkl"

    prune_mask, state_dict = retrieve_mask(folder, state_dict=True)

    if prune_mask is None:
        continue
    all_alphas = torch.cat([ts.flatten() for k in prune_mask for ts in prune_mask[k]], dim=0)
    sorted_values, _ = torch.sort(all_alphas)
    sns.histplot(sorted_values.cpu())

    print(state_dict.keys())

    vertex_pruner.load_state_dict(state_dict, strict=False)

    # # evaluate loss
    prev_edges = 0
    cand_taus = [-1.05+x*0.15 for x in range(0,20)]
    log = {"tau": [], "edges": [], "clipped_edges": [], "losses": []}

    for tau in tqdm(cand_taus):
        discrete_mask = discretize_mask(prune_mask, tau)

        edge_mask = nodes_to_mask(mask_to_nodes(discrete_mask, mask_type="nodes"), all_mlps=False)
        cpm, edges, clipped_edges, _, _ = prune_dangling_edges(edge_mask)
        if clipped_edges == prev_edges:
            continue
        prev_edges = edges
        log['tau'].append(tau)
        mask_sampler.set_mask(discrete_mask)
        kl_losses = []
        ds_iter = iter(ds_test)
        for i in tqdm(range(20)):
            batch, last_token_pos = task_ds.next_batch(tokenizer,next(ds_iter))
            with torch.no_grad():
                loss = vertex_pruner(batch, last_token_pos, timing=False)
            kl_losses.append(loss.mean().item())
        avg_loss = np.mean(kl_losses)
        log["edges"].append(edges)
        log["clipped_edges"].append(clipped_edges)
        log["losses"].append(avg_loss)
        print("Clipped edges", clipped_edges)
        print("Avg KL loss", avg_loss)

    with open(out_path, "wb") as f:
        pickle.dump(log, f)
# %%
