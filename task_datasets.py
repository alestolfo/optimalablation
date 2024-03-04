# %%
import torch
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import os
import time
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, LinePlot
import json
from pathlib import Path
from greater_than.utils import get_valid_years
from greater_than.data import YearDataset

class OWTConfig():
    def __init__(self, owt_iter):
        self.ds_iter = owt_iter

    def next_batch(self):
        batch = next(self.ds_iter)['tokens'].to(self.device)
        return batch, batch.shape[1] - 1
    
class IOIConfig():
    def __init__(self, batch_size, device, test_size=10000):
        self.batch_size = batch_size
        self.device = device
        
        ioi_ds = datasets.load_from_disk("../plausibleablation/data/ioi/ioi")

        generator = torch.Generator().manual_seed(6942)
        test_set, train_set = torch.utils.data.random_split(ioi_ds['train'], [test_size,len(ioi_ds['train']) - test_size], generator=generator)

        ioi_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        ioi_iter = cycle(iter(ioi_loader))

        self.ds_iter = ioi_iter
        self.ds_test = DataLoader(test_set, batch_size=batch_size)
    
    def get_test_set(self, tokenizer):
        return self.ds_test
    
    def init_modes(self):
        with open("modes/ioi/means_attention.pkl", "rb") as f:
            # n_layers x n_heads x d_model
            init_modes_attention = pickle.load(f)
        with open("modes/ioi/means_mlp.pkl", "rb") as f:
            # n_layers x n_heads x d_model
            init_modes_mlp = pickle.load(f)
        return init_modes_attention, init_modes_mlp

    def next_batch(self, tokenizer, test_batch=None):
        if test_batch is not None:
            b = test_batch
        else:
            b = next(self.ds_iter)
        batch = tokenizer(b['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(self.device)
        last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(self.device)).argmax(dim=-1) 
        
        # prepend bos token
        batch = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(batch.shape[0],1).to(self.device),batch], dim=1)
        return batch, last_token_pos

class GTConfig():
    def __init__(self, batch_size, device, test_size=10000):
        self.batch_size = batch_size
        self.device = device

        self.test_size = test_size
        self.years_to_sample_from = None

    def get_test_set(self, tokenizer):
        if self.years_to_sample_from is None:
            self.years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
        test_set = YearDataset(self.years_to_sample_from, self.test_size, Path("greater_than/potential_nouns.txt"), tokenizer, balanced=False, device=self.device, eos=False).good_toks
        self.ds_test = DataLoader(test_set, batch_size=self.batch_size)
        return self.ds_test

    def init_modes(self):
        with open("modes/gt/means_attention.pkl", "rb") as f:
            # n_layers x n_heads x d_model
            init_modes_attention = pickle.load(f)
        with open("modes/gt/means_mlp.pkl", "rb") as f:
            # n_layers x n_heads x d_model
            init_modes_mlp = pickle.load(f)
        return init_modes_attention, init_modes_mlp

    def next_batch(self, tokenizer, test_batch=None):
        if test_batch is not None:
            batch = test_batch
        else:
            if self.years_to_sample_from is None:
                self.years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
            batch = YearDataset(self.years_to_sample_from, self.batch_size, Path("greater_than/potential_nouns.txt"), tokenizer, balanced=False, device=self.device, eos=False).good_toks
        last_token_pos = ((batch.shape[1] - 1) * torch.ones(batch.shape[0])).int()
        # prepend bos token
        batch = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(batch.shape[0],1).to(self.device),batch], dim=1)
        return batch, last_token_pos

# NOT WORKING
class ColorConfig():
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        
        with open("color_objects/task.json") as f:
            color_ds = json.load(f)

        self.ds_iter = cycle(color_ds['examples'][:1500])
        
    def next_batch(self, tokenizer):
        batch = tokenizer(["Q: " + next(self.ds_iter)['input'] + " A: It's a" for _ in range(self.batch_size)], padding=True, return_tensors='pt')['input_ids'].to(self.device)
        last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(self.device)).argmax(dim=-1)
        return batch, last_token_pos