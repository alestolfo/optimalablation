# %%

import torch
from transformer_lens import HookedTransformer
from data import retrieve_owt_data
from itertools import cycle
import torch.optim
from fancy_einsum import einsum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
def load_model_data(model_name, batch_size=8, ctx_length=25, repeats=True, ds_name=False, device="cuda:0"):
    # device="cpu"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    if ds_name:
        owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer, ds_name=ds_name)
    else:
        owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer)
    if repeats:
        owt_iter = cycle(owt_loader)
    else:
        owt_iter = owt_loader
    return device, model, tokenizer, owt_iter

# %%

def save_hook_last_token(save_to, act, hook):
    save_to.append(act[:,-1,:])

def ablation_hook_last_token(batch_feature_idx, repl, act, hook):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    # act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)[batch_feature_idx[:,0],batch_feature_idx[:,1]]
    act = act[batch_feature_idx]
    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    # act = torch.cat([act,torch.zeros(1,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act
    # return act.repeat(features_per_batch,1,1)
    # pass

def ablation_all_hook_last_token(repl, act, hook):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    # act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)[batch_feature_idx[:,0],batch_feature_idx[:,1]]
    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    # act = torch.cat([act,torch.zeros(1,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act
    # return act.repeat(features_per_batch,1,1)
    # pass

def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def ablation_hook_attention_all_tokens(constants, bsz, activation_storage, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(constants.shape[0]):
        attentions[-start:-start+n_heads,:,i] = constants[i].clone()
        start += n_heads
    
    with torch.no_grad():
        activation_storage.append(attentions[:bsz].mean(dim=[0,1]))
    return attentions

# attentions: (batch_size + batch_size * n_samples) x seq_len x n_heads x d_model
# constants: n_heads x d_model
# prune mask: (batch_size * n_samples) x n_heads, 0 = prune, 1 = keep
def pruning_hook_attention_all_tokens(constants, prune_mask, bsz, attentions, hook):
    # N by 2. First column = batch item, second column = head idx
    prune_mask = prune_mask.unsqueeze(1).unsqueeze(-1)
    attentions[bsz:] = (1-prune_mask) * constants + prune_mask * attentions[bsz:].clone()

    # prune_idx = prune_mask.clone()
    # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
    return attentions

def tuned_lens_hook(activation_storage, tuned_lens_weights, tuned_lens_bias, act, hook):
    activation_storage.append(einsum("result activation, batch activation -> batch result", tuned_lens_weights, act[:,-1]) + tuned_lens_bias)
    return act

class LinePlot:
    def __init__(self, stat_list):
        self.stat_list = stat_list
        self.stat_book = {x: [] for x in stat_list}
        self.t = 0
    
    def add_entry(self, entry):
        for k in self.stat_book:
            if k in entry:
                self.stat_book[k].append(entry[k])
            # default behavior is flat line
            elif self.t == 0:
                self.stat_book[k].append(0)
            else:
                self.stat_book[k].append(self.stat_book[k][-1])
        self.t += 1
    
    def plot(self, series=None, step=1, start=0, end=0, agg='mean', twinx=True):
        if series is None:
            series = self.stat_list
        if end <= start:
            end = self.t
        t = [i for i in range(start, end, step)]
        ax = None
        for i,s in enumerate(series):
            if agg == 'mean':
                yvals = [np.mean(self.stat_book[s][i:i+step]) for i in range(start, end, step)]
            else:
                yvals = [self.stat_book[s][i] for i in range(start, end, step)]
            if twinx:
                colors = ["green", "blue", "red", "orange"]
                if ax is None:
                    ax = sns.lineplot(x=t, y=yvals, color=colors[i])
                else: 
                    sns.lineplot(x=t, y=yvals, ax=ax.twinx(), color=colors[i])
            else:
                ax = sns.lineplot(x=t, y=yvals, label=s)
        plt.show()
    
    def export():
        pass

