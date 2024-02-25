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

# %%

class InferenceConfig:
    def __init__(self, cfg, device, folder, init_param=-0.5):
        self.device = device
        self.batch_size = 5
        self.n_samples = 15
        self.inference_batch_size = 5
        self.folder = folder
        self.ds_iter = None
        self.lr = None
        self.lr_modes = None
        self.lamb = None
        self.record_every = 100
        self.checkpoint_every = 5

        # as in the louizos paper
        self.starting_beta = 2/3
        self.hard_concrete_endpoints = (-0.1, 1.1)

        n_heads = cfg.n_heads
        n_layers = cfg.n_layers

        self.constant_prune_mask = {
            "attn-attn": [
                # edges from attn layers into q, k, v circuits
                # first n_heads is the receiver dimension
                torch.ones((1, 3, n_heads, i, n_heads)).to(device)
                for i in range(n_layers)
            ], 
            "mlp-attn": [
                # edges from input embedding and previous MLP to q, k, v circuits
                torch.ones((1, 3, n_heads, i+1)).to(device)
                for i in range(n_layers)
            ],
            "attn-mlp": [
                # edges from attn layers into MLP and output embedding
                torch.ones((1, min(i+1, n_layers), n_heads)).to(device)
                for i in range(n_layers+1)
            ],
            "mlp-mlp": [
                # edges from input embedding and previous MLP to MLP and output embedding
                torch.ones((1, i+1)).to(device)
                for i in range(n_layers + 1)
            ]
        }

        self.layers_to_prune = []
        for layer_no in range(n_layers):
            self.layers_to_prune.append(("attn", layer_no))
            self.layers_to_prune.append(("mlp", layer_no))
        self.layers_to_prune.append(("mlp", n_layers))

        self.init_params = {
            k: [
                torch.stack(
                    [torch.ones(mask_tensor.shape[1:]) * init_param, 
                    torch.ones(mask_tensor.shape[1:]) * self.starting_beta], 
                dim=-1).to(device)
                for mask_tensor in self.constant_prune_mask[k]
            ]
            for k in self.constant_prune_mask
        }
        self.temp_adj_intv = 10
        self.temp_avg_intv = 20
        self.temp_comp_intv = 200
        self.temp_convergence_target = 2000
        self.temp_c = 0

        self.temp_momentum = 0

        self.prev_decline_rate = 0

    def temp_scheduler(self, log):
        avg_intv = self.temp_avg_intv
        comp_intv = self.temp_comp_intv
        if log.t % self.temp_adj_intv != 0:
            return self.temp_c
        
        if log.t < 1.5 * comp_intv:
            return 0            
        
        # how many % did the temperature loss decline
        decline_rate, growth_rate = log.stat_sig_growth("temp_cond", avg_intv, comp_intv)
        time_left = max(self.temp_convergence_target - (log.t - log.last_tick), 50)
        target_decline_rate = 1-np.exp(
            (-1 * np.log(np.mean(log.stat_book["temp_cond"][-20:]))) / (time_left / comp_intv))
        
        self.temp_c = max(self.temp_c, .05)

        if growth_rate > 0:
            self.temp_momentum += 1
            if self.temp_momentum * self.temp_adj_intv > time_left // 10:
                self.temp_c *= 1.05
        else:
            self.temp_momentum = max(0, self.temp_momentum - 1)
            if self.temp_momentum > 0:
                self.temp_c *= 0.95
            elif decline_rate < 0:
                self.temp_c *= 1.05
            elif decline_rate > target_decline_rate:
                self.temp_momentum = 0
                self.temp_c *= min(1 + (target_decline_rate / decline_rate - 1) / 3, 2)            
            else:
                self.temp_c *= min(1 + (target_decline_rate / decline_rate - 1) / 10, 2)   
            
        self.prev_decline_rate = decline_rate
        return self.temp_c

class OWTConfig(InferenceConfig):
    def __init__(self, owt_iter, cfg, device, folder):
        super().__init__(cfg, device, folder)

        self.ds_iter = owt_iter

    def next_batch(self):
        batch = next(self.ds_iter)['tokens'].to(self.device)
        return batch, batch.shape[1] - 1
    
class IOIConfig(InferenceConfig):
    def __init__(self, cfg, device, folder, test_size=10000, init_param=-0.5):
        super().__init__(cfg, device, folder, init_param)
        
        ioi_ds = datasets.load_from_disk("../plausibleablation/data/ioi/ioi")

        generator = torch.Generator().manual_seed(6942)
        test_set, train_set = torch.utils.data.random_split(ioi_ds['train'], [test_size,len(ioi_ds['train']) - test_size], generator=generator)

        ioi_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        ioi_iter = cycle(iter(ioi_loader))

        self.ds_iter = ioi_iter
        self.ds_test = iter(DataLoader(test_set, batch_size=self.inference_batch_size))

        # sampling params
        self.lr = 1e-1

        # modes
        self.lr_modes = 1e-3

        # reg on complexity loss
        self.lamb = 3
    
    def init_modes(self):
        with open("pruning_edges/modes/ioi/means_attention.pkl", "rb") as f:
            # n_layers x n_heads x d_model
            init_modes_attention = pickle.load(f)
        with open("pruning_edges/modes/ioi/means_mlp.pkl", "rb") as f:
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

class GTConfig(InferenceConfig):
    def __init__(self, cfg, device, folder, tokenizer, test_size=10000, init_param=-0.5):
        super().__init__(cfg, device, folder, init_param)
        self.batch_size = 15
        self.lr = 1e-1
        self.lr_modes = 1e-3
        self.lamb = 10
        self.record_every = 50
        self.years_to_sample_from = None

        self.years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
        test_set = YearDataset(self.years_to_sample_from, test_size, Path("greater_than/potential_nouns.txt"), tokenizer, balanced=False, device=self.device, eos=False).good_toks
        self.ds_test = DataLoader(test_set, batch_size=self.inference_batch_size)

    def init_modes(self):
        with open("pruning_edges/modes/owt/means_attention.pkl", "rb") as f:
            # n_layers x n_heads x d_model
            init_modes_attention = pickle.load(f)
        with open("pruning_edges/modes/owt/means_mlp.pkl", "rb") as f:
            # n_layers x n_heads x d_model
            init_modes_mlp = pickle.load(f)
        return init_modes_attention, init_modes_mlp

    def next_batch(self, tokenizer):
        if self.years_to_sample_from is None:
            self.years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
        batch = YearDataset(self.years_to_sample_from, self.batch_size, Path("greater_than/potential_nouns.txt"), tokenizer, balanced=False, device=self.device, eos=False).good_toks
        last_token_pos = (batch.shape[1] - 1) * torch.ones(batch.shape[0])
        return batch, last_token_pos

# NOT WORKING
class ColorConfig(InferenceConfig):
    def __init__(self, cfg, device, folder):
        super().__init__(cfg, device, folder)

        with open("color_objects/task.json") as f:
            color_ds = json.load(f)

        self.ds_iter = cycle(color_ds['examples'][:1500])

        self.lr = 1e-2
        self.lr_modes = 1e-3
        self.lamb = 2
    
    def next_batch(self, tokenizer):
        batch = tokenizer(["Q: " + next(self.ds_iter)['input'] + " A: It's a" for _ in range(self.batch_size)], padding=True, return_tensors='pt')['input_ids'].to(self.device)
        last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(self.device)).argmax(dim=-1)
        return batch, last_token_pos

    def temp_scheduler(self, k):
        init = 1/10
        return min(max(k-2000,0) / 20000,1) * init