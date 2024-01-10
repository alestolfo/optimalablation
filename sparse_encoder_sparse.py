# %%
import torch
import torch as t
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
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
ctx_length = 25
device, model, tokenizer, owt_loader = load_model_data(model_name, batch_size, ctx_length, ds_name="maxtli/OpenWebText-2M", repeats=False)

test_batch_size = 1000
test_ds = torch.utils.data.Subset(owt_loader.dataset,range(200000, 250000))
test_batch = test_ds[:test_batch_size]['tokens'].to(device)

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


class Superposition(t.nn.Module):
    def __init__(self, feature_dim, activation_dim):
        super().__init__()
        self.feature_weights = t.nn.Parameter(t.normal(0, 1/math.sqrt(feature_dim), (feature_dim, activation_dim)))
        self.encoder_weights = t.nn.Parameter(t.normal(0, 1/math.sqrt(feature_dim), (feature_dim, activation_dim)))
        self.bias = t.nn.Parameter(t.rand(feature_dim))
        self.relu = t.nn.ReLU()
        self.floating_mean = t.nn.Parameter(t.rand(activation_dim))
        
    def forward(self,x):
        features = (einsum("features orig, batch orig -> batch features", self.encoder_weights, x - self.floating_mean) + self.bias).relu()
        l1 = features.sum()
        l2 = features.square().sum()
        recovered = einsum("features recovered, batch features -> batch recovered", self.feature_weights, features)
        # print(self.feature_weights.isnan().sum())
        # print(self.feature_weights.shape)
        # print(recovered.shape)
        return recovered + self.floating_mean, l1, l2

# %%
feature_dim = 2000
activation_dim = 768
j = 0

sae = Superposition(feature_dim, activation_dim).to(device)
# sae.load_state_dict(torch.load(f"SAE_training/epoch_{j}.pt"))


# %%
lr = 5e-3
optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
# %%

def ev_batch(batch):
    activation_storage = []
    with torch.no_grad():
        model.run_with_hooks(
            batch,
            fwd_hooks=[(intervene_filter, 
                        partial(retrieve_activation_hook,
                                activation_storage
                        ))],
            stop_at_layer=(layer_no+1)
        )
    x = activation_storage[0][:,1:].flatten(0,1)
    # start_batch = i * int(x.shape[0]/5)
    # end_batch = (i +1)* int(x.shape[0]/5)
    recovery, l1, l2 = sae(x)
    loss = (recovery - x).square().sum()
    return loss, l1

# %%
# for j in range(1):
#     x = torch.load(f"SAE_training/activations_{j}.pt").flatten(0,2)
agg_losses = []
running_loss = [0,0]
record_freq = 25
# test_iter = islice(owt_iter,200000,250000)
test_samples_per_reading = (ctx_length - 1) * test_batch_size
train_samples_per_reading = record_freq * (ctx_length - 1) * batch_size
for i,batch in enumerate(tqdm(iter(owt_loader))):
    batch = batch['tokens'].to(device)
    optimizer.zero_grad()
    loss, l1 = ev_batch(batch)
    running_loss[0] += loss.item() / train_samples_per_reading
    running_loss[1] += .3*l1.item() / train_samples_per_reading

    if i % (-1 * record_freq) == -1 and j > 0:
        test_loss, test_l1 = ev_batch(test_batch)
        running_loss.append(test_loss.item() / test_samples_per_reading)
        running_loss.append(.3*test_l1.item() / test_samples_per_reading)
        agg_losses.append(running_loss)
        # print(running_loss)
        running_loss = [0,0]
        scheduler.step()


    loss += .7*l1
    # recovery, l1, l2 = sae(x[start_batch:end_batch])
    # loss = (recovery - x[start_batch:end_batch]).square().sum() + l1
    
    loss.backward()
    optimizer.step()

    sae.feature_weights.data /= sae.feature_weights.data.norm(dim=-1, keepdim=True)

    if i % -1000 == -1:
        torch.save(sae.state_dict(),f"SAE_training/SAE_untied_3/epoch_{j}.pt")
        if len(agg_losses) > 0:
            q=.95
            pen_qt = np.quantile([l[0] for l in agg_losses],q)
            sparsity_qt = np.quantile([l[1] for l in agg_losses],q)
            sns.lineplot(x=range(len(agg_losses)),y=[min(l[0], pen_qt) for l in agg_losses], label="reconstruction")
            sns.lineplot(x=range(len(agg_losses)),y=[min(l[1], sparsity_qt) for l in agg_losses], label="sparsity")
            sns.lineplot(x=range(len(agg_losses)),y=[min(l[2], pen_qt) for l in agg_losses], label="reconstruction_test")
            sns.lineplot(x=range(len(agg_losses)),y=[min(l[3], sparsity_qt) for l in agg_losses], label="sparsity_test")
            plt.savefig(f"SAE_training/SAE_untied_3/epoch_{j}.png")
            plt.show()
            plt.close()
            with open(f"SAE_training/SAE_untied_3/loss_graph.pkl", "wb") as f:
                pickle.dump(agg_losses, f)
        j += 1

# activation training starting from SAE initialization
# try to fix PCA training
# SAE on the model weights
# in particular this thing is fast
    


# %%
# with open("SAE_training/SAE_untied_3/loss_graph.pkl", "rb") as f:
#     agg_losses = pickle.load(f)
# # %%
# pen_qt = np.quantile([l[0] for l in agg_losses],.99)
# sns.lineplot(x=range(len(agg_losses[:1000])),y=[min(l[0], pen_qt) for l in agg_losses[:1000]], label="reconstruction")
# sns.lineplot(x=range(len(agg_losses[:1000])),y=[min(l[1], pen_qt) for l in agg_losses[:1000]], label="sparsity")
# plt.show()
# # %%


# # %%
# sae.floating_mean.norm(dim=-1)
# # %%
# bins = None
# for ep in range(5):
#     print(ep)
#     sae.load_state_dict(torch.load(f"SAE_training/SAE_untied_3/epoch_{ep}.pt"))
#     with torch.no_grad():
#         cos_sims = einsum("feature activation, feature_2 activation -> feature feature_2 ", sae.feature_weights, sae.feature_weights) - torch.eye(feature_dim).to(device)
#     if bins is None:
#         bins = np.histogram_bin_edges(cos_sims.flatten().cpu().numpy(), bins='auto')
#     sns.histplot(cos_sims.flatten().cpu().numpy(), alpha=0.2, bins=bins, label=ep)
# plt.legend()
# # sae.feature_weights.norm(dim=-1)# %%
# # %%

# act_s = []
# test_iter = iter(torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False))

# # %%
# for i in range(10):
#     with torch.no_grad():
#         model.run_with_hooks(
#             next(test_iter)['tokens'].to(device),
#             fwd_hooks=[(intervene_filter, 
#                         partial(retrieve_activation_hook,
#                                 act_s
#                         ))],
#             stop_at_layer=(layer_no+1)
#         )
# acts = torch.cat(act_s,dim=0)[:,1:]
# print(acts.mean(dim=[0,1]).norm())
# ax=sns.histplot(np.log(acts.norm(dim=-1).flatten().cpu().numpy()))
# # ax.set_xlim(left=4,right=5)

# # %%
# batched_acts = acts.flatten(0,1)
# with torch.no_grad():
#     reconstruction, _, _ = sae(batched_acts)
#     r_loss = (reconstruction - batched_acts).square().sum(dim=-1)
# sns.histplot(r_loss[:10000].sqrt().flatten().cpu().numpy())

# # %%
# normalized_acts = (acts / acts.norm(dim=-1, keepdim=True)).flatten(0,1)
# with torch.no_grad():
#     cos_sims_feat_act = einsum("feature activation, batch activation -> batch feature", sae.feature_weights, normalized_acts)

# # %%
# sns.histplot(cos_sims_feat_act[:100000].flatten().cpu().numpy())
# plt.show()

# # %%
# sns.histplot(torch.topk(cos_sims_feat_act, 10, dim=-1)[0].cpu().numpy())
# # %%
# sns.histplot(cos_sims_feat_act.relu().sum(dim=-1).cpu().numpy())

# # %%
# sns.histplot((cos_sims_feat_act > .05).sum(dim=0).log().flatten().cpu().numpy())
# # %%
# sns.histplot((cos_sims_feat_act.relu()).mean(dim=0).flatten().cpu().numpy())

# # %%

# agg_losses
# # %%

# q=1
# # pen_qt = np.quantile([l[0] for l in agg_losses],q)
# # sparsity_qt = np.quantile([l[1] for l in agg_losses],q)
# pen_qt = 350
# sparsity_qt = 350
# sns.lineplot(x=range(len(agg_losses)),y=[min(l[0], pen_qt) for l in agg_losses], label="reconstruction")
# sns.lineplot(x=range(len(agg_losses)),y=[min(l[1], sparsity_qt) for l in agg_losses], label="sparsity")
# sns.lineplot(x=range(len(agg_losses)),y=[min(l[2], pen_qt) for l in agg_losses], label="reconstruction_test")
# sns.lineplot(x=range(len(agg_losses)),y=[min(l[3], sparsity_qt) for l in agg_losses], label="sparsity_test")
# plt.show()

# # %%
