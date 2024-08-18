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
from mask_samplers.MaskSampler import MaskSampler
from utils.circuit_utils import prune_dangling_edges, discretize_mask
from utils.training_utils import update_means_variances_mixed, update_means_variances_exponential

class EdgeMaskJointSampler(MaskSampler):
    def __init__(self, pruning_cfg, node_reg=0):
        super().__init__(pruning_cfg)

        self.node_reg = node_reg
        if self.node_reg > 0:
            self.log_columns.append("node_loss")

    def node_reg_loss(self):
        n_layers = self.pruning_cfg.n_layers
        n_heads = self.pruning_cfg.n_heads
        node_losses_out = torch.zeros((n_layers, n_heads)).to(self.pruning_cfg.device)
        node_losses_in = []
        for i,ts in enumerate(self.sampling_params['attn-attn']):
            pad_layers = n_layers - i
            # 3 (dest_circ), 12 (dest head idx), i (prev n_layers), 12 (prev head idx), 2 (0-location and 1-temperature)
            layer_loss = self.complexity_loss(ts)
            node_losses_out = node_losses_out + F.pad(layer_loss, (0,0,0,pad_layers), "constant", 0).sum(dim=[0,1])
            node_losses_in.append(layer_loss.sum(dim=[0,2,3]))

        for i, ts in enumerate(self.sampling_params['attn-mlp']):
            pad_layers = max(0, n_layers - i-1)
            # i (prev n_layers), 12 (prev head idx), 2 (0-location and 1-temperature)
            layer_loss = self.complexity_loss(ts)
            node_losses_out = node_losses_out + F.pad(layer_loss, (0,0,0,pad_layers), "constant", 0)
        
        for i, ts in enumerate(self.sampling_params['mlp-attn']):
            layer_loss = self.complexity_loss(ts)
            node_losses_in[i] = node_losses_in[i] + layer_loss.sum(dim=[0,2])
        
        # derivative peaks at 1
        return F.tanh(2 * (node_losses_out + torch.stack(node_losses_in, dim=0)) / (n_layers * n_heads)) * (n_layers * n_heads) / 2       

    def forward(self):
        if not self.fix_mask:
            self.sample_mask()

        bsz = self.pruning_cfg.n_samples * self.pruning_cfg.batch_size
        prune_mask = self.sampled_mask

        # [0,1] -> {0,1} entries filtering out dangling edges
        
        with torch.no_grad():
            discrete_mask = discretize_mask(prune_mask, 0)
            filtered_mask = prune_dangling_edges(discrete_mask, bsz=bsz)
        
        for k in prune_mask:
            for i, ts in enumerate(prune_mask[k]):
                prune_mask[k][i] = ts * filtered_mask[k][i].detach()

        self.sampled_mask = prune_mask
        
        mask_loss, mask_details = self.get_mask_loss()

        if self.node_reg > 0:
            node_complexity = self.node_reg_loss().sum()
            node_reg_loss = self.node_reg * node_complexity
            mask_loss = mask_loss + node_reg_loss
            mask_details["node_loss"] = node_complexity.item()
            print("Node complexity:", node_complexity.item())

        return mask_loss, mask_details

class EdgeMaskUnifSampler(EdgeMaskJointSampler):
    def __init__(self, pruning_cfg, node_reg=0):
        super().__init__(pruning_cfg, node_reg)

        self.sampling_function = self.sample_modified_unif
        self.def_value = 0
        self.temp_scale = 1
        self.log_columns = ['complexity_loss']

        self.param_stds = {}
        self.mean_param_std = None

        self.min_window = None
        self.max_window = None
        
        self.normalize_empirical_mask = True
    
    # maximum scale = 2
    # more scale = more Unif
    def sample_modified_unif(self, unif, sampling_params, param_loc=None, dynamic_window=False):
        probs = sampling_params[...,0].sigmoid()

        if dynamic_window:
            if self.mean_param_std is None:
                print("No stds found, going to default window size")
                window_sz = self.temp_scale
            else:
                k, i = param_loc
                window_sz = (self.param_stds[k][i].squeeze(-1) / self.mean_param_std).clip(min=self.min_window,max=self.max_window)
        else:
            window_sz = self.temp_scale
        
        window_sz = (window_sz * probs * (1 - probs)).detach()

        # scale invariance
        probs = window_sz * probs - (window_sz - 1) * probs.detach()

        return 1-((unif - probs) / window_sz + 0.5).clamp(0,1)
    
    def update_param_vars(self, adam_vars, clip_q=0.01):
        # adam vars is in a long list
        all_adam_stds = []
        i = 0
        for k in self.sampling_params:
            self.param_stds[k] = []
            for j, ts in enumerate(self.sampling_params[k]):
                assert adam_vars[i].shape == ts.shape
                adam_std = adam_vars[i].sqrt()
                self.param_stds[k].append(adam_std)
                all_adam_stds.append(adam_std.flatten())
                i += 1
        avg_std = torch.cat(all_adam_stds, dim=0)
        avg_std = avg_std.clip(max=avg_std.quantile(1-clip_q))
        self.mean_param_std = avg_std.mean()

    def sample_bernoulli(self, unif, sampling_params, param_loc=None):
        with torch.no_grad():
            probs = sampling_params[...,0].sigmoid()
            return (unif < probs.detach()) * 1

    def complexity_loss(self, sampling_params):
        return sampling_params[...,0].sigmoid()
    
    def get_mask_loss(self):
        all_sampling_params = self.get_sampling_params()

        # alphas already logged
        complexity_loss = self.complexity_loss(all_sampling_params)
        mask_loss = self.pruning_cfg.lamb * complexity_loss.sum()

        with torch.no_grad():
            print("Complexity:", complexity_loss.sum().item(), "out of", complexity_loss.nelement())

        mask_details = {                
            "complexity_loss": complexity_loss.mean().item() if self.complexity_mean else complexity_loss.sum().item(),
        }
        return mask_loss, mask_details
    
    def forward(self):
        return super().forward()

    def record_state(self, j):
        all_sampling_params = self.get_sampling_params()

        sns.histplot(torch.cat([
            ts.flatten() for k in self.sampled_mask for ts in self.sampled_mask[k]
        ], dim=0).detach().flatten().cpu(), log_scale=(False, True))
        plt.savefig(f"{self.pruning_cfg.folder}/mask{j}.png")
        plt.close()

        sns.histplot(x=all_sampling_params.sigmoid().detach().flatten().cpu(), bins=100, log_scale=(False, True))
        plt.savefig(f"{self.pruning_cfg.folder}/params-probs{j}.png")
        plt.close()
    
    def clip_grad(self, bound):
        grad_norms = []
        for k in self.sampling_params:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.sampling_params[k], max_norm=float('inf'))
            grad_norms.append(grad_norm.item())
            torch.nn.utils.clip_grad_norm_(self.sampling_params[k], max_norm=bound)
        return grad_norms