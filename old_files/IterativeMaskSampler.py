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
from MaskSampler import MaskSampler
from ..utils.circuit_utils import prune_dangling_edges, discretize_mask

class EdgeMaskIterativeSampler(MaskSampler):
    def __init__(self, pruning_cfg, reverse=True):
        super().__init__(pruning_cfg, complexity_mean=True)
        self.layers_to_prune = list(reversed(pruning_cfg.layers_to_prune)) if reverse else pruning_cfg.layers_to_prune
        self.layer_idx = -1
        self.prune_mask = self.pruning_cfg.constant_prune_mask
        self.sampled_mask = None
        self.next_layer()
        self.compute_edges()
    
    def next_layer(self):
        self.layer_idx += 1
        if self.layer_idx >= len(self.layers_to_prune):
            return False
        component_type, cur_layer = self.layers_to_prune[self.layer_idx]
        self.component_type = component_type
        self.cur_layer = cur_layer
        return True
    
    def get_sampling_params(self):
        return torch.cat([self.sampling_params[f"attn-{self.component_type}"][self.cur_layer].flatten(start_dim=0,end_dim=-2), self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer].flatten(start_dim=0,end_dim=-2)], dim=0)
    
    def compute_edges(self):
        self.total_edges = np.sum([(ts > 0).sum().item() for k in self.prune_mask for ts in self.prune_mask[k]])
    
    def freeze_params(self, tau=0):
        if self.prune_mask[f"attn-{self.component_type}"][self.cur_layer].nelement() > 0:
            self.prune_mask[f"attn-{self.component_type}"][self.cur_layer] = ((self.sampling_params[f"attn-{self.component_type}"][self.cur_layer][...,0] > tau) * 1).unsqueeze(0)
        self.prune_mask[f"mlp-{self.component_type}"][self.cur_layer] = ((self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer][...,0] > tau) * 1).unsqueeze(0)

        self.compute_edges()

    def forward(self):
        prune_mask = self.prune_mask
        if self.prune_mask[f"attn-{self.component_type}"][self.cur_layer].nelement() > 0:
            attn_unif = torch.rand((
                self.pruning_cfg.n_samples * self.pruning_cfg.batch_size, 
                *self.sampling_params[f"attn-{self.component_type}"][self.cur_layer].shape[:-1]
            )).to(self.pruning_cfg.device)
            prune_mask[f"attn-{self.component_type}"][self.cur_layer] = self.sampling_function(
                attn_unif, 
                self.sampling_params[f"attn-{self.component_type}"][self.cur_layer]
            ).clone()

        mlp_unif = torch.rand((
            self.pruning_cfg.n_samples * self.pruning_cfg.batch_size, 
            *self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer].shape[:-1]
        )).to(self.pruning_cfg.device)
        prune_mask[f"mlp-{self.component_type}"][self.cur_layer] = self.sampling_function(
            mlp_unif, 
            self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer]
        ).clone()

        self.sampled_mask = prune_mask
        
        return self.get_sampling_params()
    
    def take_snapshot(self, j):
        metadata_path = f"{self.pruning_cfg.folder}/mask-status{j}.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump((self.layer_idx, self.prune_mask), f)

    def load_snapshot(self):
        metadata_path = f"{self.pruning_cfg.folder}/mask-status.pkl"
        with open(metadata_path, "rb") as f:
            layer_idx, mask = pickle.load(f)
        self.prune_mask = mask
        self.layer_idx = layer_idx - 1
        self.next_layer()