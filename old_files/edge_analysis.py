# %%
import torch
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
from copy import deepcopy
import time
from itertools import cycle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.MaskConfig import IOIConfig, GTConfig, cache_hook_all_tokens, pruning_edge_attention_hook_all_tokens, pruning_edge_mlp_hook_all_tokens, get_cache_hooks, sample_prune_mask
from training_utils import load_model_data, LinePlot
# %%

# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

kl_loss = torch.nn.KLDivLoss(reduction="none")

# %%

# settings
cache_compressed_attn = True
if not cache_compressed_attn:
    model.cfg.use_attn_result = True

folder="pruning_edges_auto/gt_iter"
PARAMS = GTConfig(model.cfg, device, folder)

# %%
epoch = 48
with open(f"{folder}/train_{epoch}.pkl", "rb") as f:
#     # n_layers x n_heads x d_model
    pruning_values = pickle.load(f)

with open(f"{folder}/modes_{epoch}.pkl", "rb") as f:
    # n_layers x n_heads x d_model
    modal_values = pickle.load(f)

# %%

my_df = []
for cur_layer, ts in enumerate(pruning_values["attn-attn"]):
    for idx, circ in enumerate(["q","k","v"]):
        for head_idx in range(ts.shape[1]):
            for layer_idx in range(ts.shape[2]):
                for prev_head_idx in range(ts.shape[3]):
                    my_df.append({"type": "attn-attn", "cur_layer": cur_layer, "circ": circ, "head_idx": head_idx, "prev_layer": layer_idx, "prev_head_idx": prev_head_idx, "wt": ts[idx, head_idx, layer_idx, prev_head_idx, 0].item()})
for cur_layer, ts in enumerate(pruning_values["mlp-attn"]):
    for idx, circ in enumerate(["q","k","v"]):
        for head_idx in range(ts.shape[1]):
            for layer_idx in range(ts.shape[2]):
                    my_df.append({"type": "mlp-attn", "cur_layer": cur_layer, "circ": circ, "head_idx": head_idx, "prev_layer": layer_idx, "wt": ts[idx, head_idx, layer_idx, 0].item()})
for cur_layer, ts in enumerate(pruning_values["attn-mlp"]):
    for layer_idx in range(ts.shape[0]):
        for prev_head_idx in range(ts.shape[1]):
            my_df.append({"type": "mlp-attn", "cur_layer": cur_layer, "prev_layer": layer_idx, "prev_head_idx": prev_head_idx, "wt": ts[layer_idx, prev_head_idx, 0].item()})
for cur_layer, ts in enumerate(pruning_values["mlp-mlp"]):
    for layer_idx in range(ts.shape[0]):
        my_df.append({"type": "mlp-mlp", "cur_layer": cur_layer, "prev_layer": layer_idx, "wt": ts[layer_idx, 0].item()})

# %%
        
my_df = pd.DataFrame(my_df)


# %%

constant_prune_mask = deepcopy(PARAMS.constant_prune_mask)

for k in constant_prune_mask:
    for x in range(len(constant_prune_mask[k])):
        # print(constant_prune_mask[k][x].shape)
        constant_prune_mask[k][x] *= (pruning_values[k][x][...,0] > 2)

# %%

# sample pruned heads independently from batch, or use same pruned heads for each batch item?
# currently using the former

# %%

circs = ["q", "k", "v"]
attention_cache = []
mlp_cache = []
cache_hooks = get_cache_hooks(cache_compressed_attn, n_layers, attention_cache, mlp_cache)

# resid_mid_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out"

attention_in_filter = lambda layer_no, circ, name: name == f"blocks.{layer_no}.hook_{circ}_input"
mlp_in_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_in"
final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

post_bias = torch.stack([model.blocks[layer_no].attn.b_O.clone().detach() for layer_no in range(n_layers)], dim=0)

if cache_compressed_attn:
    W_O = torch.stack([model.blocks[layer_no].attn.W_O.clone().detach() for layer_no in range(n_layers)], dim=0)

for param in model.parameters():
    param.requires_grad = False

# %%
lp = LinePlot(['kl_loss', 'step_size', 'mode_step_size'])
lp_2 = LinePlot(['av_alpha', 'complexity_loss', 'temp_loss'])

all_kl_losses = []

with torch.no_grad():
    no_batches = 0
    no_checkpoints = 0
    while no_batches < 100:
        batch, last_token_pos = PARAMS.next_batch(tokenizer)
        last_token_pos = last_token_pos.int()

        # sample prune mask
        prune_mask = constant_prune_mask

        attention_cache = []
        mlp_cache = []
        
        pruned_output = model.run_with_hooks(
            batch, 
            fwd_hooks=[
                *cache_hooks,
                
                # patch attention (recompute O if compressed)
                *[(partial(attention_in_filter, layer_no, circ), 
                    partial(pruning_edge_attention_hook_all_tokens, 
                            W_O[:layer_no] if cache_compressed_attn else None, 
                            # prune_mask[layer_no][j], 
                            [prune_mask["attn-attn"][layer_no][:,j] if layer_no > 0 else None, prune_mask["mlp-attn"][layer_no][:,j]], 
                            modal_values[0][:layer_no], 
                            modal_values[1][:layer_no+1], 
                            attention_cache, 
                            mlp_cache, 
                            post_bias[:layer_no].sum(dim=0))) 
                for layer_no in range(n_layers) for j, circ in enumerate(circs)],

                # patch MLP (recompute O if compressed)
                *[(partial(mlp_in_filter, layer_no), 
                    partial(pruning_edge_mlp_hook_all_tokens, 
                            W_O[:layer_no+1] if cache_compressed_attn else None, 
                            # prune_mask[layer_no][-1], 
                            [prune_mask["attn-mlp"][layer_no], prune_mask["mlp-mlp"][layer_no]], 
                            modal_values[0][:layer_no+1], 
                            modal_values[1][:layer_no+1], 
                            attention_cache,
                            mlp_cache, 
                            post_bias[:layer_no+1].sum(dim=0))) 
                for layer_no in range(n_layers)],

                # patch MLP (recompute O if compressed)
                (final_embed_filter, 
                    partial(pruning_edge_mlp_hook_all_tokens, 
                            W_O if cache_compressed_attn else None, 
                            # prune_mask[-1], 
                            [prune_mask["attn-mlp"][-1], prune_mask["mlp-mlp"][-1]], 
                            modal_values[0], 
                            modal_values[1], 
                            attention_cache,
                            mlp_cache, 
                            post_bias.sum(dim=0))),
        ])

        pruned_output = pruned_output[torch.arange(pruned_output.shape[0]),last_token_pos].log_softmax(dim=-1)

        orig_output = model(batch)
        orig_output = orig_output[torch.arange(orig_output.shape[0]), last_token_pos].log_softmax(dim=-1)

        kl_losses = kl_loss(pruned_output, orig_output.exp()).sum(dim=-1)
        # io_loss = target_results - ablated_results

        print(kl_losses.mean())

        all_kl_losses.append(kl_losses)

# %%
# %%
