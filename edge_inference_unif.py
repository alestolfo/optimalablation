# %%
import torch
from sys import argv
import numpy as np 
from tqdm import tqdm
import torch.optim
import seaborn as sns
import pickle
from EdgePruner import EdgePruner
from MaskSampler import ConstantMaskSampler, EdgeMaskUnifSampler
from MaskConfig import EdgeInferenceConfig
from task_datasets import IOIConfig, GTConfig
from circuit_utils import discretize_mask, prune_dangling_edges, retrieve_mask, mask_to_edges, mask_to_nodes
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

folder=f"pruning_edges_auto/ioi_edges_unif/{reg_lamb}"
out_path=f"pruning_edges_auto/ioi_edges_unif/report/{str(reg_lamb).replace('.', '-')}.pkl"

batch_size = 10
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, batch_size=batch_size)
pruning_cfg.lamb = reg_lamb
pruning_cfg.initialize_params_probs(1)
task_ds = IOIConfig(batch_size, device)
ds_test = task_ds.get_test_set(tokenizer)

for param in model.parameters():
    param.requires_grad = False

# %%
# mask_sampler = ConstantMaskSampler()
# edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
# edge_pruner.add_cache_hooks()
# edge_pruner.add_patching_hooks()

# %%

mask_sampler = EdgeMaskUnifSampler(pruning_cfg, node_reg=5e-3)
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

# %%
prune_mask, state_dict = retrieve_mask(folder, state_dict=True)
all_alphas = torch.cat([ts.flatten() for k in prune_mask for ts in prune_mask[k]], dim=0)
sorted_values, _ = torch.sort(all_alphas)
sns.histplot(sorted_values.cpu())

print(state_dict.keys())

edge_pruner.load_state_dict(state_dict, strict=False)
# %%
# gpu_requeue = True
# lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue, pretrained_folder=None)

# take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)
# %%
# if prune_retrain and edge_pruner.log.t == 0:
#     edge_pruner.log.mode = "prune"
#     edge_pruner.log.cur_counter = 0

with torch.no_grad():
    max_batches = 10000
    for no_batches in tqdm(range(edge_pruner.log.t, max_batches)):

        plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
        checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

        batch, last_token_pos = task_ds.next_batch(tokenizer)
        last_token_pos = last_token_pos.int()

        # sample prune mask
        graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
        loss = edge_pruner(batch, last_token_pos, graph_suffix)

        prev_alphas = mask_sampler.get_sampling_params()[:,0].detach().clone()
        prev_modes = edge_pruner.get_modes().detach().clone()

        break

# %%

discrete_mask = discretize_mask(prune_mask, 0)
cpm, edges, clipped_edges, _, _ = prune_dangling_edges(discrete_mask)

# %%
mask_sampler.set_mask(cpm)
kl_losses = []
ds_iter = iter(ds_test)
for i in tqdm(range(20)):
    batch, last_token_pos = task_ds.next_batch(tokenizer,next(ds_iter))
    with torch.no_grad():
        loss = edge_pruner(batch, last_token_pos, timing=False)
    kl_losses.append(loss.mean().item())

# mask_to_edges(discrete_mask)

# %%

# %%

# # evaluate loss
prev_edges = 0
cand_taus = [-2+x*0.4 for x in range(0,10)]
log = {"tau": [], "edges": [], "clipped_edges": [], "losses": []}

for tau in tqdm(cand_taus):
    discrete_mask = discretize_mask(prune_mask, tau)
    cpm, edges, clipped_edges, _, _ = prune_dangling_edges(discrete_mask)
    if clipped_edges == prev_edges:
        continue
    prev_edges = edges
    log['tau'].append(tau)
    mask_sampler.set_mask(cpm)
    kl_losses = []
    ds_iter = iter(ds_test)
    for i in tqdm(range(20)):
        batch, last_token_pos = task_ds.next_batch(tokenizer,next(ds_iter))
        with torch.no_grad():
            loss = edge_pruner(batch, last_token_pos, timing=False)
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
