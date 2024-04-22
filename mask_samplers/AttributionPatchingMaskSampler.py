import torch
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
import math
from functools import partial
import torch.optim
import time
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from circuit_utils import prune_dangling_edges, discretize_mask

# for attribution patching
class AttributionPatchingMaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg):
        super().__init__()

        self.use_temperature = False
        self.log_columns = []

        n_layers = pruning_cfg.n_layers
        n_heads = pruning_cfg.n_heads
        device = pruning_cfg.device

        bsz = pruning_cfg.batch_size

        self.sampling_params = torch.nn.ParameterDict({
            "attn": torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((n_heads,)).to(device)) 
                for _ in range(n_layers)
            ]),
            "mlp": torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((1,)).to(device)) 
                for _ in range(n_layers)
            ])
        })

        self.sampled_mask = {}
        for k in self.sampling_params:
            self.sampled_mask[k] = []
            for ts in enumerate(self.sampling_params[k]):
                self.sampled_mask[k].append(torch.ones((bsz, *ts.shape)).to(device) * ts)

    def forward(self):
        return 0, {}

    def record_state(self, j):
        pass

# for direct mean ablation
class SingleComponentMaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg):
        super().__init__()

        self.use_temperature = False
        self.log_columns = []

        n_layers = pruning_cfg.n_layers
        device = pruning_cfg.device

        total_heads = n_layers * pruning_cfg.n_heads

        bsz = pruning_cfg.batch_size

        if pruning_cfg.n_samples != total_heads:
            raise Exception("In patching mode, we need to patch all heads")

        # [bsz, n_layers, n_heads]
        attn_mask = (torch.ones((total_heads, total_heads)) - torch.eye(total_heads))
        attn_mask = attn_mask.unflatten(1, (n_layers, -1)).to(device)
        self.sampling_params = {
            "attn": [
                attn_mask[:, i]
                for i in range(n_layers)
            ],
            "mlp": [
                torch.ones((total_heads,)).to(device) 
                for i in range(n_layers)
            ]
        }

        self.sampled_mask = {}
        for k in self.sampling_params:
            self.sampled_mask[k] = []
            for ts in self.sampling_params[k]:
                self.sampled_mask[k].append((torch.ones((bsz, *ts.shape)).to(device) * ts).flatten(start_dim=0, end_dim=1))

    def forward(self):
        return 0, {}

    def record_state(self, j):
        pass

# for gradient sampling
class MultiComponentMaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg, prop_sample=0.1):
        super().__init__()
        
        self.sampled_mask = None

        self.use_temperature = False
        self.log_columns = []

        self.pruning_cfg = pruning_cfg

        self.n_layers = pruning_cfg.n_layers
        self.n_heads = pruning_cfg.n_heads
        self.device = pruning_cfg.device

        self.prop_sample = prop_sample

        self.mask_perturb = torch.nn.ParameterDict({
            "attn": torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros((self.n_heads,)).to(self.device)) 
                for i in range(self.n_layers)
            ]),
            "mlp": torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros((1,)).to(self.device)) 
                for i in range(self.n_layers)
            ])
        })

    def forward(self):
        bsz = self.pruning_cfg.bsz * self.pruning_cfg.n_samples

        total_heads = self.n_layers * self.n_heads
        sampled_heads = math.ceil(self.prop_sample * total_heads)

        # select random subset
        ref_idx = torch.arange(bsz).unsqueeze(-1).repeat(1, sampled_heads)
        _, top_k_idx = torch.rand((bsz, total_heads)).topk(sampled_heads, dim=-1)

        attn_mask = torch.ones((bsz, total_heads))
        attn_mask[ref_idx.flatten(), top_k_idx.flatten()] = 0
        attn_mask = attn_mask + (1-attn_mask) * torch.rand_like(attn_mask)
        attn_mask = attn_mask.unflatten(1, (self.n_layers, -1)).to(self.device)

        fixed_mask = {
            "attn": [
                attn_mask[:, i]
                for i in range(self.n_layers)
            ],
            "mlp": [
                torch.ones((bsz,1)).to(self.device) 
                for i in range(self.n_layers)
            ]
        }

        self.sampled_mask = {}
        for k in self.mask_perturb:
            self.sampled_mask[k] = []
            for i, ts in enumerate(self.mask_perturb[k]):
                self.sampled_mask[k].append(fixed_mask[k][i] + torch.ones(bsz, *ts.shape).to(self.device) * ts)

        return 0, {}

    def record_state(self, j):
        pass