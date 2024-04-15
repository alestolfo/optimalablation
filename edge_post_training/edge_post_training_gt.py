# %%
# import torch
import os
from sys import argv
from tqdm import tqdm
import argparse
import torch.optim
from EdgePruner import EdgePruner
from mask_samplers.MaskSampler import ConstantMaskSampler
from utils.MaskConfig import EdgeInferenceConfig
from task_datasets import IOIConfig, GTConfig
from circuit_utils import discretize_mask, prune_dangling_edges, retrieve_mask
from training_utils import load_model_data, LinePlot

# %%

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lamb',
                        help='regularization constant')
    parser.add_argument('-s', '--subfolder',
                        help='where to save stuff')
    parser.add_argument('-t', '--tau',
                        help='threshold')
    args = parser.parse_args()
    reg_lamb = args.lamb
    subfolder = args.subfolder
    tau = float(args.tau)
except:
    # reg_lamb = 2e-4
    subfolder = None
    tau = -1

try:
    reg_lamb = float(reg_lamb)
except:
    pass

if reg_lamb is None:
    reg_lamb = 2e-4

gpu_requeue = True
# reset_optim = 1000
if subfolder is not None:
    folder=f"pruning_edges_auto/{subfolder}/{reg_lamb}"
else:
    folder=f"pruning_edges_auto/gt_edges_unif/{reg_lamb}"

# %%
    

print(folder)
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

batch_size = 75
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, batch_size=batch_size)
# pruning_cfg.lamb = reg_lamb
pruning_cfg.n_samples = 1

task_ds = GTConfig(batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

prune_mask, state_dict = retrieve_mask(folder, state_dict=True)

if os.path.exists(f"{folder}/fit_nodes_{tau}.pth"):
    state_dict = torch.load(f"{folder}/fit_nodes_{tau}.pth")

edge_pruner.load_state_dict(state_dict, strict=False)

discrete_mask = discretize_mask(prune_mask, tau)
cpm, edges, clipped_edges, _, _ = prune_dangling_edges(discrete_mask)
mask_sampler.set_mask(cpm)

modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%
max_batches = 6000
for no_batches in tqdm(range(edge_pruner.log.t, max_batches)):

    modal_optimizer.zero_grad()

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    loss = edge_pruner(batch, last_token_pos, timing=False)

    edge_pruner.log.add_entry({
        "kl_loss": loss.mean().item()
    })

    loss.mean().backward()
    modal_optimizer.step()

    if no_batches % -100 == -1:
        print(f"Saving {folder}/fit_modes_{tau}.pth")
        torch.save({"modal_attention": edge_pruner.modal_attention, "modal_mlp": edge_pruner.modal_mlp}, f"{folder}/fit_modes_{tau}.pth")
        edge_pruner.log.plot(["kl_loss"], mv=100, save=f"{folder}/fit_modes_{tau}.png")
# %%
