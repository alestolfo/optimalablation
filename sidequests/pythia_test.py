# %%
import torch
from transformer_lens import HookedTransformer
from data import retrieve_owt_data
from itertools import cycle
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

# %%
model_name = "EleutherAI/pythia-70m-deduped"
# model_name = "gpt2-small"

# device="cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)

# %%

tokenizer = model.tokenizer
batch_size = 8
max_context_length = 35
owt_loader = retrieve_owt_data(batch_size, max_context_length, tokenizer)
owt_iter = cycle(owt_loader)

# %%

lg = model(next(owt_iter)['tokens'])
lg[:,:,(lg < -20).nonzero()[:,2].unique()].shape
sns.histplot(lg[:,:,(lg < -20).nonzero()[:,2].unique()].flatten().detach().cpu().numpy())

# %%
unembed_norms = model.W_U.norm(dim=0)

# %%
sns.histplot(unembed_norms[torch.min(unembed_norms > 11, unembed_norms < 30).nonzero()].detach().cpu().numpy())
# %%
sns.histplot(lg[:,:,(unembed_norms < 12.5).nonzero()].flatten().detach().cpu().numpy())

# %%
sns.histplot(lg[:,:,(unembed_norms > 12.5).nonzero()].flatten().detach().cpu().numpy())

# %%
sns.histplot(unembed_norms[(lg < -60).nonzero()[:,2].unique()].detach().cpu().numpy())
# %%
