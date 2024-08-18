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
from einops import repeat
from utils.circuit_utils import prune_dangling_edges, discretize_mask

# for direct mean ablation
class SingleComponentMaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg):
        super().__init__()

        self.log_columns = []

        self.n_components = torch.cat([ts.flatten() for k in pruning_cfg.init_params for ts in pruning_cfg.init_params[k]], dim=0).shape[0]

        component_mask = (torch.ones((self.n_components, self.n_components)) - torch.eye(self.n_components)).to(pruning_cfg.device)

        
        self.sampled_mask = {}
        start = 0
        for k in pruning_cfg.init_params:
            self.sampled_mask[k] = []
            for ts in pruning_cfg.init_params[k]:
                n = ts.nelement()
                # [batch_size * n_components, n]
                # CONVENTION: since we repeat the batch tokens n_samples times, the correct unflattened shape for embeddings is [n_samples, batch_size, seq, d_model]
                # t: total components, c: components in this layer
                mask = repeat(component_mask[:, start:(start + n)], "t c -> (t b) c", b=pruning_cfg.batch_size)
                mask = mask.reshape((pruning_cfg.batch_size * self.n_components, *ts.shape[:-1]))

                self.sampled_mask[k].append(mask)
                start += n
        
    def forward(self):
        return 0, {}

    def record_state(self, j):
        pass

# for gradient sampling
class MultiComponentMaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg, prop_sample=0.1):
        super().__init__()
        
        self.sampled_mask = None

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
        attn_mask = attn_mask + (1-attn_mask) * (
            torch.rand_like(attn_mask) + 
            torch.stack(self.mask_perturb['attn'], dim=0).flatten()
        )
        attn_mask = attn_mask.unflatten(1, (self.n_layers, -1)).to(self.device)

        self.sampled_mask = {
            "attn": [
                attn_mask[:, i]
                for i in range(self.n_layers)
            ],
            "mlp": [
                torch.ones((bsz,1)).to(self.device) 
                for i in range(self.n_layers)
            ]
        }
        return 0, {}

    def record_state(self, j):
        pass