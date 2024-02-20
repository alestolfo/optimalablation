# %%

import torch

from transformer_lens import HookedTransformer
from itertools import cycle
import torch.optim
from fancy_einsum import einsum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import partial
# import pickle 
import datasets 
from tqdm import tqdm
from training_utils import load_model_data
# import torch
# from einops import rearrange
from torch.utils.data import DataLoader, random_split
# %%
import einops
import numpy as np

# %%
model_name = "gpt2-small"
folder = "pruning/pruning_modes_gt"
batch_size = 15
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
# model.cfg.use_attn_result = True

# inverse probe setting

# %%

attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"
def pruning_hook_attention_all_tokens(activation_storage, bsz, attentions, hook):
    print('hi')
    activation_storage.append(attentions.detach().clone())
    # N by 2. First column = batch item, second column = head idx
    # prune_mask = prune_mask.unsqueeze(1).unsqueeze(-1)
    # attentions[bsz:] = (1-prune_mask) * constants + prune_mask * attentions[bsz:].clone()

    # prune_idx = prune_mask.clone()
    # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
    return attentions

# %%

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-3

# %%
model.eval()

with torch.no_grad():
    batch = next(owt_iter)['tokens']
    activation_storage = []
    model.run_with_hooks(
        batch,             
        fwd_hooks=[
                (partial(attention_points_filter, layer_no), 
                   partial(pruning_hook_attention_all_tokens,
                           activation_storage,
                           batch_size)
                ) for layer_no in range(n_layers)
            ]
        )
# %%
