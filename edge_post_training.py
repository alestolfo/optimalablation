# %%
# import torch
import os
from sys import argv
from tqdm import tqdm
import torch.optim
import pickle
from pruners.EdgePruner import EdgePruner
from mask_samplers.MaskSampler import ConstantMaskSampler
from utils.MaskConfig import EdgeInferenceConfig
from utils.task_datasets import get_task_ds
from utils.circuit_utils import discretize_mask, prune_dangling_edges, retrieve_mask, edges_to_mask
from utils.training_utils import load_model_data, LinePlot, load_args    

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
args = load_args("pruning", 2e-4)
folder, reg_lamb, dataset, tau, run_name, ablation_type = args["folder"], args["lamb"], args["dataset"], args["tau"], args["name"], args["desc"]
print("Folder", folder)
print("Lamb", reg_lamb)
print("Dataset", dataset)
print("Ablation type", ablation_type)

gpu_requeue = True
# reset_optim = 1000

batch_size = 75
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, batch_size=batch_size)
pruning_cfg.n_samples = 1

task_ds = get_task_ds(dataset, batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

if run_name == "acdc" or run_name == "eap":
    p_folder = folder.rsplit("/", 1)[0]
    acdc_edges_file=f"{p_folder}/edges_{reg_lamb}.pth"
    edge_list = torch.load(acdc_edges_file)
    discrete_mask = edges_to_mask(edge_list)
else:
    prune_mask = retrieve_mask(folder)
    discrete_mask = discretize_mask(prune_mask, tau)

cpm, c_e, clipped_e, _, _ = prune_dangling_edges(discrete_mask)
print(c_e)
print(clipped_e)
mask_sampler.set_mask(cpm)

# if os.path.exists(f"{folder}/fit_nodes_{tau}.pth"):
#     if os.path.exists(f"{folder}/fit_loss_log.pkl"):
#         with open(f"{folder}/fit_loss_log.pkl", "rb") as f:
#             log = pickle.load(f)
#         edge_pruner.log = log
#     state_dict = torch.load(f"{folder}/fit_nodes_{tau}.pth")
#     edge_pruner.load_state_dict(state_dict, strict=False)

modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=5 * pruning_cfg.lr_modes, weight_decay=0)

# %%
max_batches = 10000
for no_batches in tqdm(range(edge_pruner.log.t, max_batches)):

    modal_optimizer.zero_grad()

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    loss = edge_pruner(batch, last_token_pos, timing=False)
    loss.backward()
    modal_optimizer.step()

    if no_batches % -100 == -1:
        print(f"Saving {folder}/fit_modes_{tau}.pth")
        torch.save({"modal_attention": edge_pruner.modal_attention, "modal_mlp": edge_pruner.modal_mlp}, f"{folder}/fit_modes_{tau}.pth")
        
        with open(f"{folder}/fit_loss_log.pkl", "wb") as f:
            pickle.dump(edge_pruner.log, f)
        
        edge_pruner.log.plot(["kl_loss"], mv=100, save=f"{folder}/fit_modes_{tau}.png")
# %%
