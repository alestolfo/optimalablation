# %%
import torch
from tqdm import tqdm
from sys import argv
import torch.optim
import os
from utils.training_utils import load_model_data, LinePlot
from utils.circuit_utils import retrieve_mask, discretize_mask, get_ioi_nodes, nodes_to_vertex_mask, nodes_to_mask, prune_dangling_edges
from mask_samplers.MaskSampler import ConstantMaskSampler
from pruners.VertexPruner import VertexPruner
from pruners.EdgePruner import EdgePruner
from utils.MaskConfig import EdgeInferenceConfig
from utils.task_datasets import IOIConfig, GTConfig

# %%

model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.train()
model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%

batch_size=75
pruning_cfg = EdgeInferenceConfig(model.cfg, device, None, init_param=0, batch_size=batch_size)
pruning_cfg.n_samples = 1

task_ds = IOIConfig(batch_size, device)

for param in model.parameters():
    param.requires_grad = False

# %%
vertex_sampler = ConstantMaskSampler()
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), vertex_sampler)

ioi_nodes = get_ioi_nodes()
vertex_mask = nodes_to_vertex_mask(ioi_nodes)
vertex_sampler.set_mask(vertex_mask)

edge_sampler = ConstantMaskSampler()
edge_pruner = EdgePruner(model, pruning_cfg, task_ds.init_modes(), edge_sampler)

edge_mask = nodes_to_mask(ioi_nodes)
discrete_mask = discretize_mask(edge_mask, 0.5)
cpm, edges, clipped_edges, _, _ = prune_dangling_edges(discrete_mask)
edge_sampler.set_mask(cpm)

# %%

for no_batches in tqdm(range(1)):

    with torch.no_grad():
        batch, last_token_pos = task_ds.next_batch(tokenizer)
        vertex_pruner.add_patching_hooks()
        vertex_outputs = vertex_pruner(batch, last_token_pos, timing=False, return_output=True)
        model.reset_hooks()

        edge_pruner.add_cache_hooks()
        edge_pruner.add_patching_hooks()
        edge_outputs = edge_pruner(batch, last_token_pos, timing=False, return_output=True)
        model.reset_hooks()

        print((vertex_outputs - edge_outputs).abs().max())

# %%
