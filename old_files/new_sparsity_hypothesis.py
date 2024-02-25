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
batch_size = 200
ctx_length = 25
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size, ctx_length)

# inverse probe setting
layer_no = 3
num_features = 3000
activation_dim = 768
# features_per_batch = 50 * batch_size

# %%
# learning hyperparameters
convergence_tol = 5e-2
similarity_tol = .05
lr_act = 1e-1
lr_feat = 1e-2
sparse_lambda = 3e-2
updates_per_batch = 200
relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

intervene_filter = lambda name: name == f"blocks.{layer_no}.hook_resid_post"

# %%

# init_features = torch.rand((num_features, activation_dim)).to(device)
# init_features /= init_features.norm(dim=-1, keepdim=True)

# folder = "v3"
# with open(f"init_sae/{folder}/feature_{0}.pkl", "rb") as f:
#     init_features = (pickle.load(f)).to(device)

sae = UntiedEncoder(num_features, activation_dim).to(device)
sae.load_state_dict(torch.load(f"SAE_training/SAE_rerun_untied/epoch_{0}.pt"))

init_features = sae.feature_weights.detach()
floating_mean = sae.floating_mean.detach()
# # feature_directions = torch.normal(0,1,(num_features, activation_dim)).to(device)
init_features = init_features / init_features.norm(dim=-1, keepdim=True)

feature_param = torch.nn.Parameter(init_features)
feature_optimizer = torch.optim.SGD([feature_param], lr=lr_feat, weight_decay=0)

# %%
def sparsify_activations(batch, feature_param, feature_optimizer, lp):

    def update_activations(target_probs, activation_param, activation_optimizer, activation_scheduler):
        activation_optimizer.zero_grad()

        sparse_activation = einsum("features recovered, batch features -> batch recovered", sae.feature_weights, activation_param.relu())

        cur_log_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[(intervene_filter, 
                        partial(ablation_all_hook_last_token,
                                sparse_activation)
                        )]
        )[:,-1].log_softmax(dim=-1)
        # [:,-1].softmax(dim=-1)

        kl_losses = kl_loss(cur_log_probs, target_probs).sum(dim=-1)

        # feature_similarities = ((einsum("batch activation, feature activation -> batch feature", activation_param - floating_mean, feature_param) / (activation_param - floating_mean).norm(dim=-1, keepdim=True)) - 0.1).relu()
        feature_similarities = (activation_param - 0.05).relu()

        loss = (kl_losses + sparse_lambda * feature_similarities.sum(dim=-1)).sum()

        loss.backward()

        prev_activations = activation_param.detach().clone()

        activation_optimizer.step()
        activation_scheduler.step()

        avg_step_size = (activation_param.detach() - prev_activations).norm(dim=-1).mean()
        # .norm(dim=-1).mean()

        return {'act_step_size': avg_step_size.item(), 'kl_loss': kl_losses.mean().item(), 'sparsity_loss': feature_similarities.sum(dim=-1).mean().item()}

    with torch.no_grad():
        # save the original activations
        cur_activations = []

        # -1 gives last token
        target_probs = model.run_with_hooks(batch, fwd_hooks=[(intervene_filter, partial(save_hook_last_token, cur_activations))])[:,-1].softmax(dim=-1)

        cur_activations = cur_activations[0]

    for param in model.parameters():
        param.requires_grad = False

    # linear combination of features
    feature_combo = (einsum("features orig, batch orig -> batch features", sae.encoder_weights, cur_activations - floating_mean) + sae.bias).relu()

    activation_param = torch.nn.Parameter(feature_combo)
    activation_optimizer = torch.optim.Adam([activation_param], lr=lr_act, weight_decay=0)
    activation_scheduler = torch.optim.lr_scheduler.StepLR(activation_optimizer, step_size=50, gamma=0.9)

    avg_step_size = 1
    convergence_steps = 0
    while avg_step_size > convergence_tol:
        # print(act_stats['act_step_size'], act_stats['sparsity_loss'])
        act_stats = update_activations(target_probs, activation_param, activation_optimizer, activation_scheduler)

        lp.add_entry(act_stats)

        convergence_steps += 1

        if convergence_steps % -50 == -1:
            lp.plot(['kl_loss', 'sparsity_loss', 'act_step_size'])
            print(act_stats)
        
        avg_step_size = act_stats['act_step_size']
    
    for i in tqdm(range(updates_per_batch)):
        feature_optimizer.zero_grad()
        act_stats = update_activations(target_probs, activation_param, activation_optimizer, activation_scheduler)

        # print(feature_param)
        
        prev_features = feature_param.detach().clone()
        feature_optimizer.step()
        avg_step_size = (feature_param.detach() - prev_features).norm(dim=-1).mean()

        # print(feature_param)

        break
        act_stats['feat_step_size'] = avg_step_size.item()
        lp.add_entry(act_stats)

        if i % -50 == -1:
            lp.plot(['kl_loss', 'sparsity_loss', 'act_step_size', 'feat_step_size'], start=convergence_steps // 3)
            print(act_stats)


# %%
i = 0
while i < 1000:
    batch = next(owt_iter)['tokens']
    lp = LinePlot(['act_step_size', 'feat_step_size', 'kl_loss', 'sparsity_loss'])
    sparsify_activations(batch, feature_param, feature_optimizer, lp)

    break

    if i % -10 == -1:
        lp.plot(step=updates_per_batch)
    i += 1

    
 # %%
