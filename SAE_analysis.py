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
convergence_tol = 1e-4
similarity_tol = .05
lr_act = 1e-4
lr_feat = 1e-5
sparse_lambda = 1e-2
updates_per_batch = 100
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

# %%

def retrieve_activations():
    activations = []
    model.eval()
    for i in tqdm(range(100)):
        batch = next(owt_iter)['tokens']
        with torch.no_grad():
            model.run_with_hooks(
                batch,
                fwd_hooks=[(intervene_filter, 
                            partial(save_hook_last_token,
                                    activations)
                )],
                stop_at_layer=4
            )
    return torch.cat(activations, dim=0)

activations = retrieve_activations()
# %%
feature_similarities = (einsum("batch activation, feature activation -> batch feature", activations - floating_mean, init_features) / (activations - floating_mean).norm(dim=-1, keepdim=True)).detach()

feature_similarities.shape

# %%

sns.histplot(feature_similarities[:1000].detach().flatten().cpu().numpy())

# %%

# top k most liked features
sns.histplot(torch.topk(feature_similarities[:10000], 5, dim=-1)[0].cpu().numpy())

# %%
sns.histplot(feature_similarities[:,torch.topk(feature_similarities.relu().mean(dim=0), 10, dim=-1)[1]].cpu().numpy())

# %%

# sns.histplot(feature_similarities[:,torch.topk(feature_similarities.relu().mean(dim=0) * -1, 100, dim=-1)[1]].cpu().numpy())

# %%

sns.histplot(feature_similarities.mean(dim=0).cpu().numpy())



# %%

sns.histplot((feature_similarities-.1).relu().sum(dim=1).cpu().numpy())

# %%

floating_mean.norm()
# %%
pair_similarities = (einsum("feature1 activation, feature2 activation -> feature1 feature2", init_features, init_features)-torch.eye(init_features.shape[0]).to(device)).detach()

# %%
sns.histplot(pair_similarities[:100].flatten().detach().cpu().numpy())

# %%

_, S, v = torch.pca_lowrank(init_features, 768)
# %%
S
# %%
v.norm(dim=-1)
# %%
