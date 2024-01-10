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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, save_hook_last_token, ablation_hook_last_token

# %%


# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 50
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

# inverse probe setting
layer_no = 6
pca_dimension = 400
activation_dim = 768
lr=1e-4

intervene_filter = lambda name: name == f"blocks.{layer_no}.hook_resid_post"

# %%

encoder_mtrx = torch.nn.Parameter(torch.normal(0,1,(pca_dimension, activation_dim)).to(device))
decoder_mtrx = torch.nn.Parameter(torch.normal(0,1,(pca_dimension, activation_dim)).to(device))
centroid = torch.nn.Parameter(torch.normal(0,1,(activation_dim,)).to(device))
optimizer = torch.optim.SGD([encoder_mtrx, decoder_mtrx], lr=lr, weight_decay=0)
kl_loss = torch.nn.KLDivLoss(reduction="none")
ce_loss = torch.nn.CrossEntropyLoss()
# %%

def pca_hook_last_token(encoder, decoder, centroid, act, hook):
    return einsum("d_compression d_model, d_compression d_out, batch seq d_model -> batch seq d_out", encoder, decoder, act) + centroid

# %%
losses = []
for i in tqdm(range(100000)):
    batch = next(owt_iter)['tokens'].to(device)

    optimizer.zero_grad()

    orig_probs = model(batch)[:,-1].softmax(dim=-1)
    cur_logits = model.run_with_hooks(
        batch, 
        fwd_hooks=[(intervene_filter, 
                    partial(pca_hook_last_token,
                            encoder_mtrx,
                            decoder_mtrx,
                            centroid
                    ))]
    )


    loss = ce_loss(cur_logits[:,:-1].permute(0,2,1), batch[:,1:])
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        folder = "pca"
        with open(f"outputs/{folder}/feature_{i}.pkl", "wb") as f:
            pickle.dump(encoder_mtrx.data.detach(), f)
        with open(f"outputs/{folder}/updates_{i}.pkl", "wb") as f:
            pickle.dump(decoder_mtrx.data.detach(), f)
        with open(f"outputs/{folder}/av_e_{i}.pkl", "wb") as f:
            pickle.dump(centroid.data.detach(), f)
        with open(f"outputs/{folder}/loss_graph.pkl", "wb") as f:
            pickle.dump(losses, f)

# %%
folder="pca"
i=5000
with open(f"outputs/{folder}/feature_{i}.pkl", "rb") as f:
    encoder_mtrx = pickle.load(f)
with open(f"outputs/{folder}/updates_{i}.pkl", "rb") as f:
    decoder_mtrx = pickle.load(f)
with open(f"outputs/{folder}/av_e_{i}.pkl", "rb") as f:
    centroid = pickle.load(f)
with open(f"outputs/{folder}/loss_graph.pkl", "rb") as f:
    loss_graph = pickle.load(f)



# %%
for i in range(20):
    batch = next(owt_iter)['tokens'].to(device)
    with torch.no_grad():
        orig_logits = model(batch)[:,:-1]
        logits = model.run_with_hooks(
            batch, 
            fwd_hooks=[(intervene_filter, 
                        partial(pca_hook_last_token,
                                encoder_mtrx,
                                decoder_mtrx,
                                centroid
                        ))]
        )[:,:-1]
        print(logits.shape)


        loss = ce_loss(orig_logits.permute(0,2,1), batch[:,1:])
        print(loss)
        loss = ce_loss(logits.permute(0,2,1), batch[:,1:])
        print(loss)

# %%
sns.lineplot(loss_graph)
# %%
