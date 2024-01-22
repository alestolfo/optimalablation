# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
from encoders import UntiedEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, ablation_hook_copy_all_tokens, ablation_hook_attention_all_tokens, LinePlot

# %%

# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 20
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
lr = 1e-3

# # learning hyperparameters
# convergence_tol = 1e-4
# similarity_tol = .05
# lr_act = 1e-4
# lr_feat = 1e-5
# updates_per_batch = 100
# relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"

# %%
modal_attentions = [torch.nn.Parameter(torch.rand(n_heads, head_dim)) for _ in range(n_layers)]
modal_optimizer = torch.optim.SGD(modal_attentions, lr=lr, weight_decay=1e-3)

for param in model.parameters():
    param.requires_grad = False

# %%
    
i = 0
while i < 1000:
    batch = next(owt_iter)['tokens']
    modal_optimizer.zero_grad()

    model_results = model.run_with_hooks(
            batch,
            fwd_hooks=[
                *[(partial(attention_points_filter, layer_no), 
                   partial(ablation_hook_attention_all_tokens,
                            modal_attentions[layer_no],
                            batch_size)
                    ) for layer_no in range(n_layers)],
                *[(partial(resid_points_filter, layer_no), 
                   partial(ablation_hook_copy_all_tokens,
                           batch_size)
                    ) for layer_no in range(n_layers)]]
    )[:,-1].softmax(dim=-1)

    # batch_size x vocab_size
    target_results = model_results[:batch_size]

    # (n_layers * n_heads) x batch_size x vocab_size
    ablated_results = model_results[batch_size:].unflatten(0, (batch_size,n_layers * n_heads)).permute((1,0,2,3))

    # might need to fix this ???
    kl_losses = kl_loss(ablated_results.log(), target_results).sum(dim=-1)

    loss = kl_losses.sum()
    loss.backward()

    modal_optimizer.step()
 