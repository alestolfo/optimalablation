# %%

import torch
from transformer_lens import HookedTransformer
from data import retrieve_owt_data
from itertools import cycle
import torch.optim
import matplotlib.pyplot as plt

# %%
def load_model_data(model_name, batch_size=8, ctx_length=25, device="cuda:0"):
    # device="cpu"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer)
    owt_iter = cycle(owt_loader)
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
