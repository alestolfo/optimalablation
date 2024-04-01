# %%
import torch
from tqdm import tqdm
from sys import argv
import torch.optim
import os
from training_utils import load_model_data, LinePlot
from circuit_utils import retrieve_mask, discretize_mask, get_ioi_nodes, nodes_to_vertex_mask
from MaskSampler import ConstantMaskSampler
from VertexPruner import VertexPruner
from MaskConfig import VertexInferenceConfig
from task_datasets import IOIConfig, GTConfig

# %%

model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.train()
# model.cfg.use_attn_result = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
# settings
try:
    reg_lamb = float(argv[1])
    tau = float(argv[2])
except:
    reg_lamb=1e-3
    tau = -1

manual=False

if manual:
    folder=f"pruning_vertices_auto/ioi/manual"
    tau = 0.5
else:
    folder=f"pruning_vertices_auto/ioi_with_mlp/{reg_lamb}"

batch_size=75
pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=0, batch_size=batch_size)
pruning_cfg.lamb = reg_lamb

task_ds = IOIConfig(batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
vertex_pruner.add_patching_hooks()

if manual:
    ioi_nodes = get_ioi_nodes()
    prune_mask = nodes_to_vertex_mask(ioi_nodes)
    mask_sampler.set_mask(prune_mask)
else:
    prune_mask, state_dict = retrieve_mask(folder, state_dict=True)
    if os.path.exists(f"{folder}/fit_nodes_{tau}.pth"):
        state_dict = torch.load(f"{folder}/fit_nodes_{tau}.pth")
        print(state_dict)
    vertex_pruner.load_state_dict(state_dict, strict=False)
    discrete_mask = discretize_mask(prune_mask, tau)
    mask_sampler.set_mask(discrete_mask)

modal_optimizer = torch.optim.AdamW([vertex_pruner.modal_attention, vertex_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)

# %%

max_batches = 6000
for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):

    modal_optimizer.zero_grad()

    batch, last_token_pos = task_ds.next_batch(tokenizer)
    loss = vertex_pruner(batch, last_token_pos, timing=False)

    vertex_pruner.log.add_entry({
        "kl_loss": loss.mean().item()
    })

    loss.mean().backward()
    modal_optimizer.step()

    if no_batches % -100 == -1:
        torch.save({"modal_attention": vertex_pruner.modal_attention, "modal_mlp": vertex_pruner.modal_mlp}, f"{folder}/fit_modes_{tau}.pth")
        vertex_pruner.log.plot(["kl_loss"], mv=100, save=f"{folder}/fit_modes_{tau}.png")
# %%
