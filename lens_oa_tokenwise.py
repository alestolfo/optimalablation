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
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, save_hook_last_token, LinePlot
from utils.lens_utils import LensExperiment

# %%
sns.set()
folder="results/lens/oa_tokenwise"
shared_bias = False

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 200
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 2e-3

kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

# prior_bias = [
#     model.blocks[i].attn.b_O.clone() for i in range(n_layers)
# ]
prior_bias = [
    torch.randn_like(model.blocks[i].attn.b_O) for i in range(n_layers)
]

attn_bias = [
    # torch.nn.Parameter(torch.ones((i+1, d_model)).to(device)) for i in range(n_layers)
    torch.nn.Parameter(prior_bias[i].to(device)) for i in range(n_layers)
]

lp = LinePlot([*[f"kl_loss_{k}" for k in range(n_layers)], 'step_size'])

lens_optimizer = torch.optim.AdamW(attn_bias, lr=lr, weight_decay=0)

for param in model.parameters():
    param.requires_grad = False

for p in attn_bias:
    p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))

exp = LensExperiment(model, owt_iter, {}, device, pretrained=False)
exp.all_lens_bias['oa_tokenwise'] = attn_bias

# %%
def save_hook_first_last(activation_storage, act, hook):
    # [bsz, 2, d_model]
    activation_storage.append(torch.stack([act[:,0],act[:,-1]], dim=1))

def apply_oa_tokenwise_lens(self, activation_storage, seq_len):
    bsz = activation_storage[0].shape[0]
    attn_bias = self.all_lens_bias['oa_tokenwise']

    resid = []
    for layer_no in range(self.model.cfg.n_layers):
        if layer_no > 0:
            # [layer_no, batch, d_model]
            resid = torch.cat([resid, activation_storage[layer_no]], dim=0)
        else:
            resid = activation_storage[layer_no]

        # attn_bias[layer_no]: [d_model,]
        assert resid.shape == ((layer_no + 1) * bsz, 2, d_model)
        null_resid = torch.ones(((layer_no + 1) * bsz, seq_len, 1)).to(device) * attn_bias[layer_no]
        null_resid[:,0] = resid[:,0]
        null_resid[:,-1] = resid[:,-1]

        normalized_resid_pre = self.model.blocks[layer_no].ln1(null_resid)
        attn_out = self.model.blocks[layer_no].attn(
            query_input=normalized_resid_pre,
            key_input=normalized_resid_pre,
            value_input=normalized_resid_pre
            # attention_mask
        )

        resid_mid = resid + torch.stack([attn_out[:,0], attn_out[:,-1]], dim=1)
        normalized_resid_mid = self.model.blocks[layer_no].ln2(resid_mid)
        mlp_out = self.model.blocks[layer_no].mlp(normalized_resid_mid)
        resid = resid_mid + mlp_out
    
    # [n_layers, batch, d_model]
    resid = self.model.ln_final(resid)

    modal_lens_probs = self.model.unembed(resid)

    # [batch, n_layers, d_vocab]
    modal_lens_probs = modal_lens_probs.softmax(dim=-1)[:,-1].unflatten(0, (n_layers, bsz)).permute((1,0,2))

    return modal_lens_probs

# %%    
for i in tqdm(range(lp.t, 50000)):
    batch = next(owt_iter)['tokens']
    lens_optimizer.zero_grad()
    
    activation_storage = []

    with torch.no_grad():
        model_probs = model.run_with_hooks(
                batch,
                fwd_hooks=[
                    *[(partial(resid_points_filter, layer_no), 
                    partial(save_hook_first_last, activation_storage),
                        ) for layer_no in range(n_layers)],
                    ]
        )[:,-1].softmax(dim=-1).unsqueeze(1)
        
    modal_lens_probs = apply_oa_tokenwise_lens(exp, activation_storage, batch.shape[-1])

    kl_losses = kl_loss(modal_lens_probs.log(), model_probs).sum(dim=-1).mean(dim=0)
    loss = kl_losses.sum()

    loss.backward()

    prev_weights = torch.cat(attn_bias, dim=0).detach()

    lens_optimizer.step()

    step_sz = (torch.cat(attn_bias, dim=0)-prev_weights).abs().sum()
    lp.add_entry({
        "step_size": step_sz.item(), 
        **{f"kl_loss_{k}": kl_losses[k].item() for k in range(n_layers)}
    })
    
    # lens_scheduler.step()

    if math.isnan(lp.stat_book["step_size"][-1]):
        break

    if i % -500 == -1:
        lp.plot(subplots=3, save=f"{folder}/train.png", twinx=False, mv=20)
        with open(f"{folder}/lens_bias.pkl", "wb") as f:
            pickle.dump(attn_bias, f)
# %%

