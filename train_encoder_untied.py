# %%
import torch
import torch as t
from transformer_lens import HookedTransformer
from encoders import UntiedEncoder
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
from training_utils import load_model_data, save_hook_last_token, ablation_hook_last_token, LinePlot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%

model_name = "gpt2-small"
batch_size = 100
ctx_length = 25
device, model, tokenizer, owt_loader = load_model_data(model_name, batch_size, ctx_length, ds_name="maxtli/OpenWebText-2M", repeats=False)

test_batch_size = 1000
test_ds = torch.utils.data.Subset(owt_loader.dataset,range(200000, 250000))
test_batch = test_ds[:test_batch_size]['tokens'].to(device)

model.eval()

# inverse probe setting
layer_no = 3
pca_dimension = 400
activation_dim = 768
lr=1e-4

intervene_filter = lambda name: name == f"blocks.{layer_no}.hook_resid_post"

# %%

def retrieve_activation_hook(activation_storage, act, hook):
    activation_storage.append(act)

# %%
    
feature_dim = 4000
activation_dim = 768
j = 0

sae = UntiedEncoder(feature_dim, activation_dim).to(device)
# sae.load_state_dict(torch.load(f"SAE_training/epoch_{j}.pt"))


# %%
lr = 1e-3
optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.98)
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

running_loss = [0,0]
record_freq = 25

lp = LinePlot(['reconstruct_train', 'reconstruct_test', 'sparsity_train', 'sparsity_test'])

lp_2 = LinePlot(['step_sz', 'mode_magnitude'])
# test_iter = islice(owt_iter,200000,250000)
test_samples_per_reading = (ctx_length - 1) * test_batch_size
train_samples_per_reading = record_freq * (ctx_length - 1) * batch_size

# %%
for i,batch in enumerate(tqdm(iter(owt_loader))):
    batch = batch['tokens'].to(device)
    optimizer.zero_grad()
    loss, l1 = ev_batch(batch)
    running_loss[0] += loss.item() / train_samples_per_reading
    running_loss[1] += l1.item() / train_samples_per_reading
    loss += .7*l1
    # recovery, l1, l2 = sae(x[start_batch:end_batch])
    # loss = (recovery - x[start_batch:end_batch]).square().sum() + l1
    
    loss.backward()

    if i % (-1 * record_freq) == -1:
        prev_weights = sae.feature_weights.clone().detach()
 
        optimizer.step()
        step_sz = (sae.feature_weights.detach() - prev_weights).norm(dim=-1).mean()

        test_loss, test_l1 = ev_batch(test_batch)

        lp.add_entry({ 
            'reconstruct_train': running_loss[0] / record_freq, 
            'reconstruct_test': test_loss.item() / test_samples_per_reading,'sparsity_train': running_loss[1] / record_freq,
            'sparsity_test': test_l1.item() / test_samples_per_reading, 
        })
        lp_2.add_entry({
            'step_sz': step_sz.item(), 
            'mode_magnitude': sae.floating_mean.norm().item() 
        })
        # print(running_loss)
        running_loss = [0,0]
        scheduler.step()
    else:
        optimizer.step()

    sae.feature_weights.data /= sae.feature_weights.data.norm(dim=-1, keepdim=True)

    if i % -1000 == -1:
        torch.save(sae.state_dict(),f"SAE_training/SAE_rerun_untied_2/epoch_{j}.pt")
        lp.plot(start= min(lp.t // 2,100), save=f"SAE_training/SAE_rerun_untied_2/epoch_{j}.png")
        lp_2.plot(save=f"SAE_training/SAE_rerun_untied_2/epoch_{j}_plot2.png")
        j += 1

# activation training starting from SAE initialization
# try to fix PCA training
# SAE on the model weights
# in particular this thing is fast
# %%
