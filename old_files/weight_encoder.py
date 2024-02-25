# %%
import torch
import torch as t
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from encoders import UntiedEncoder
from einops import rearrange
from itertools import islice
import math
from functools import partial
import torch.optim
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, save_hook_last_token, ablation_hook_last_token

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%


# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 100
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size, ds_name="maxtli/OpenWebText-2M", repeats=False)
# 
model.eval()

# inverse probe setting
layer_no = 6
pca_dimension = 400
activation_dim = 768
lr=1e-4

intervene_filter = lambda name: name == f"blocks.{layer_no}.hook_resid_post"

# %%

def retrieve_activation_hook(activation_storage, act, hook):
    activation_storage.append(act)

# %%



# activation_storage = []
# j = 0
# for i,batch in enumerate(tqdm(owt_iter)):
#     batch = batch['tokens'].to(device)
#     with torch.no_grad():
#         model.run_with_hooks(
#             batch, 
#             fwd_hooks=[(intervene_filter, 
#                         partial(retrieve_activation_hook,
#                                 activation_storage
#                         ))],
#             stop_at_layer=(layer_no+1)
#         )    
#     # with open(f"SAE_training/activations_{i}.pkl", "wb") as f:
#     if i % 10 == 9:
#         torch.save(torch.stack(activation_storage,dim=0), f"SAE_training/activations_{j}.pt")
#         j += 1
#         activation_storage = []

# %%

# PCA and sparse autoencoders

attn_params = []
mlp_params = []
attn_in = ["W_K", "W_Q", "W_V"]
for k, param in model.named_parameters():
    print(k)
    elts = k.split(".")
    if len(elts) < 3:
        continue
    if elts[2] == "attn" and elts[3] in attn_in:
        attn_params.append(param.permute(0,2,1).flatten(0,1))
    if elts[2] == "mlp" and elts[3] == "W_in":
        mlp_params.append(param.permute(1,0))
# %%
feature_dim = 100
sae = UntiedEncoder(feature_dim, activation_dim).to(device)

# %%

attn_params = torch.cat(attn_params, dim=0)
mlp_params = torch.cat(mlp_params, dim=0)
# %%

# all_params = torch.cat([attn_params, mlp_params, model.unembed.W_U.permute(1,0)], dim=0).detach()
all_params = torch.cat([attn_params, mlp_params], dim=0).detach()
dataset = torch.utils.data.TensorDataset(attn_params)

# %%
lr = 5e-3
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
agg_losses = []

# %%

# %%
for epoch in tqdm(range(100)):
    running_loss = [0,0]
    for batch in iter(dataloader):
        optimizer.zero_grad()
        # 0 is the activations
        recovery, l1, l2 = sae(batch[0])
        loss = (recovery - batch[0]).square().sum()
        running_loss[0] += loss.item() / len(dataset)
        running_loss[1] += l1.item() / len(dataset)
        loss += .05 * l1
        loss.backward()
        optimizer.step()
        sae.feature_weights.data /= sae.feature_weights.data.norm(dim=-1, keepdim=True)
    scheduler.step()
    agg_losses.append(running_loss)
    if epoch % -10 == -1:
        q = .9
        pen_qt = np.quantile([l[0] for l in agg_losses],q)
        sparsity_qt = np.quantile([l[1] for l in agg_losses],q)
        sns.lineplot(x=range(len(agg_losses)),y=[min(l[0], pen_qt) for l in agg_losses], label="reconstruction")
        sns.lineplot(x=range(len(agg_losses)),y=[min(l[1], sparsity_qt) for l in agg_losses], label="sparsity")
        plt.show()
    
    
# %%

sns.lineplot(x=range(len(agg_losses)),y=[min(l[0], pen_qt) for l in agg_losses], label="reconstruction")
sns.lineplot(x=range(len(agg_losses)),y=[min(l[1], sparsity_qt) for l in agg_losses], label="sparsity")
plt.show()

# %%
torch.save(sae.state_dict(),f"SAE_training/all_weights_3.pt")

# %%

sns.histplot(all_params.norm(dim=-1).cpu().numpy())

# %%
