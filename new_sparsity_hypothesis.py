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
from training_utils import load_model_data, save_hook_last_token, ablation_all_hook_last_token, LinePlot

# %%


# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 20
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size, device="cpu")

# inverse probe setting
layer_no = 3
num_features = 2000
activation_dim = 768
# features_per_batch = 50 * batch_size

# learning hyperparameters
convergence_tol = 1e-4
similarity_tol = .05
lr_act = 1e-4
lr_feat = 1e-5
updates_per_batch = 100
relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

intervene_filter = lambda name: name == f"blocks.{layer_no}.hook_resid_post"

# %%

init_features = torch.rand((num_features, activation_dim))
init_features /= init_features.norm(dim=-1, keepdim=True)

feature_param = torch.nn.Parameter(init_features)
feature_optimizer = torch.optim.SGD([feature_param], lr=lr_feat, weight_decay=0)

# %%
def sparsify_activations(batch, feature_param, feature_optimizer):

    def update_activations(target_probs, activation_param, activation_optimizer):
        activation_optimizer.zero_grad()

        cur_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[(intervene_filter, 
                        partial(ablation_all_hook_last_token,
                                activation_param)
                        )]
        )[:,-1].softmax(dim=-1)

        kl_losses = kl_loss(cur_probs.log(), target_probs).sum(dim=-1)

        feature_similarities = einsum("batch activation, feature activation -> batch feature", activation_param, feature_param).sum(dim=-1) / activation_param.norm(dim=-1)

        loss = (kl_losses + feature_similarities).sum()

        loss.backward()

        prev_activations = activation_param.detach()
        activation_optimizer.step()

        avg_step_size = (activation_param.detach() - prev_activations).norm(dim=-1).mean()

        return avg_step_size

    with torch.no_grad():
        # save the original activations
        cur_activations = []

        # -1 gives last token
        target_probs = model.run_with_hooks(batch, fwd_hooks=[(intervene_filter, partial(save_hook_last_token, cur_activations))])[:,-1].softmax(dim=-1)

        cur_activations = cur_activations[0]

    for param in model.parameters():
        param.requires_grad = False

    activation_param = torch.nn.Parameter(cur_activations)
    activation_optimizer = torch.optim.Adam([activation_param], lr=lr_act, weight_decay=0)

    lp = LinePlot([''])

    avg_step_size = 1
    while avg_step_size > convergence_tol:
        update_activations(target_probs, activation_param, activation_optimizer)

    # for i in range(updates_per_batch):
    #     feature_optimizer.zero_grad()

    # feature_optimizer.step()

# %%
i = 0
while i < 1000:
    batch = next(owt_iter)['tokens']
    sparsify_activations(batch, feature_param, feature_optimizer)
    i += 1