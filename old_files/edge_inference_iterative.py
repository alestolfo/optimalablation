# %%
import torch
from sys import argv
import numpy as np 
from tqdm import tqdm
import torch.optim
import os
import seaborn as sns
import pickle
import glob
from EdgePruner import EdgePruner
from mask_samplers.MaskSampler import ConstantMaskSampler
from utils.MaskConfig import EdgeInferenceConfig
from task_datasets import IOIConfig, GTConfig
from circuit_utils import discretize_mask, prune_dangling_edges, retrieve_mask
from training_utils import load_model_data, LinePlot

# %%
# load model
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
# settings
try:
    reg_lamb = float(argv[1])
except:
    reg_lamb=1e-4

folder=f"pruning_edges_auto/ioi_iter"
out_path=f"pruning_edges_auto/ioi_iter/pre_training.pkl"

batch_size = 50
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, batch_size=batch_size)
pruning_cfg.lamb = reg_lamb
pruning_cfg.n_samples = 1
task_ds = IOIConfig(batch_size, device)
ds_test = task_ds.get_test_set(tokenizer)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True, ablation_backward=True)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

# %%
log = {"lamb": [], "tau": [], "losses": [], "edges": [], "clipped_edges": []}

for lamb_path in glob.glob(f"{folder}/*"):
    lamb = lamb_path.split("/")[-1]
    print(lamb)
    try:
        float(lamb)
    except:
        continue

    prune_mask, state_dict = retrieve_mask(lamb_path, state_dict=True)
    all_alphas = torch.cat([ts.flatten() for k in prune_mask for ts in prune_mask[k]], dim=0)
    sorted_values, _ = torch.sort(all_alphas)
    sns.histplot(sorted_values.cpu())

    discrete_mask = discretize_mask(prune_mask, -1)
    discrete_mask, edges, clipped_edges, _, _ = prune_dangling_edges(discrete_mask)

    print(state_dict.keys())

    edge_pruner.load_state_dict(state_dict, strict=False)
    mask_sampler.set_mask(discrete_mask)

    ds_iter = iter(ds_test)
    kl_losses = []

    for i in tqdm(range(20)):
        batch, last_token_pos = task_ds.next_batch(tokenizer, next(ds_iter))
        with torch.no_grad():
            loss = edge_pruner(batch, last_token_pos, timing=False)
            kl_losses.append(loss.mean().item())

    avg_loss = np.mean(kl_losses)
    log["lamb"].append(lamb)
    log["tau"].append(0)
    log["edges"].append(edges)
    log["clipped_edges"].append(clipped_edges)
    log["losses"].append(avg_loss)
    print("Clipped edges", clipped_edges)
    print("Avg KL loss", avg_loss)

with open(f"{folder}/pre_training.pkl", "wb") as f:
    pickle.dump(log, f)


# %%
