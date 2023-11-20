# %%

from transformer import DemoTransformer, Config
import torch
from transformer_lens import HookedTransformer
from data import retrieve_owt_data
from itertools import cycle
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
import math
from functools import partial
from torch.optim import AdamW
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)

# %%
my_model = DemoTransformer(Config(debug=False))
# %%
my_model.load_state_dict(model.state_dict(), strict=False)
my_model.to(device)
# %%

tokenizer = model.tokenizer
batch_size = 8
max_context_length = 35
owt_loader = retrieve_owt_data(batch_size, max_context_length, tokenizer)
owt_iter = cycle(owt_loader)

# %%
def ablation_hook(act):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    # act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)
    
    # [batch_feature_idx[:,0],batch_feature_idx[:,1]]

    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    # act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    act = torch.cat([act,torch.zeros(4,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act

# %%
batch = next(owt_iter)['tokens']
# %%
x = my_model(batch, hook=(2,ablation_hook))
# %%
y = my_model(batch)
# %%
torch.sum(torch.abs(x[0]-y[0]))
# %%
