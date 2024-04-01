# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
import datasets
from itertools import cycle
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
from sys import argv
import math
from functools import partial
import torch.optim
import time
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, LinePlot
from task_datasets import OWTConfig, IOIConfig, GTConfig
# %%
# import sys
# del sys.modules['task_datasets']
# %%
# dataset settings

folder = "oca/owt"

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 100
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)
model.train()
# model.cfg.use_attn_result = True

# %%
task_ds = OWTConfig(owt_iter, device)
# task_ds = IOIConfig(batch_size, device)
# task_ds = GTConfig(batch_size, device)

# %%
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model

kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_post_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_post"

# %%

def save_activation_hook(last_token_pos, activation_storage, activations, hook):
    activation_storage.append(activations[torch.arange(batch_size),last_token_pos])

# %%
baseline = []
for i in tqdm(range(1000)):
    # modify depending on the dataset

    batch, last_token_pos = task_ds.next_batch()
    
    with torch.no_grad():
        activation_storage = []
        fwd_hooks = [*[(partial(resid_post_filter, layer_no), 
                    partial(save_activation_hook,
                            last_token_pos,
                            activation_storage)
                        ) for layer_no in range(n_layers)]]

        model_results = model.run_with_hooks(
                batch,
                fwd_hooks=fwd_hooks
        )
    activation_storage = torch.stack(activation_storage, dim=1)
    baseline.append(activation_storage)
# %%
baseline = torch.cat(baseline, dim=0)
print(baseline.shape)

# %%
with open(f"{folder}/baseline_resid.pkl", "wb") as f:
    pickle.dump(baseline,f)

# %%
