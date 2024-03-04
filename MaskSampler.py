
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
import pickle
from circuit_utils import prune_dangling_edges, discretize_mask

class MaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg):
        super().__init__()

        self.pruning_cfg = pruning_cfg

        self.sampling_params = torch.nn.ParameterDict({
            k: torch.nn.ParameterList([
                torch.nn.Parameter(p_init) for p_init in pruning_cfg.init_params[k]
            ]) for k in pruning_cfg.init_params
        })

        self.sampled_mask = None

        for param in self.parameters():
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))
        
    def get_sampling_params(self):
        # returns N x 2 tensor
        return torch.cat([ts.flatten(start_dim=0, end_dim=-2) if len(ts.shape) > 1 else ts.unsqueeze(0) for k in self.sampling_params for ts in self.sampling_params[k]], dim=0)
    
    def sample_prune_mask(self, unif, sampling_params):
        # back prop against log alpha
        endpts = self.pruning_cfg.hard_concrete_endpoints
        concrete = (((.001+unif).log() - (1-unif).log() + sampling_params[...,0])/(sampling_params[...,1].relu()+.001)).sigmoid()

        hard_concrete = ((concrete + endpts[0]) * (endpts[1] - endpts[0])).clamp(0,1)

        # n_layers x (total_samples = batch_size * n_samples) x n_heads
        return hard_concrete
        
    def fix_nans(self):
        fixed = True
        with torch.no_grad():
            sampling_params = self.get_sampling_params()
            
            nancount = sampling_params.isnan().sum()

            if nancount > 0:
                print("NANs", nancount)
                for k in self.sampling_params:
                    for ts in self.sampling_params[k]:
                        ts[ts[:,1].isnan().nonzero()[:,0],1] = 2/3
                        if ts.isnan().sum() > 0:
                            err = False
        return fixed

    # beta and alpha should be same shape as x, or broadcastable
    # def f_concrete(x, beta, alpha):
    #     return ((x.log() - (1-x).log()) * beta - alpha.log()).sigmoid()

    def forward(self):
        bsz = self.pruning_cfg.n_samples * self.pruning_cfg.batch_size
        prune_mask = {}
        for k in self.sampling_params:
            prune_mask[k] = []
            for i in range(len(self.sampling_params[k])):
                # if sampling_params[k][i].nelement() == 0:
                #     prune_mask[k].append(None)
                #     continue
                unif = torch.rand((bsz, *self.sampling_params[k][i].shape[:-1])).to(self.pruning_cfg.device)
                prune_mask[k].append(self.sample_prune_mask(unif, self.sampling_params[k][i]))

        self.sampled_mask = prune_mask

        return self.get_sampling_params()

    def take_snapshot(self, j):
        pass

    def load_snapshot(self):
        pass

class EdgeMaskJointSampler(MaskSampler):
    def __init__(self, pruning_cfg):
        super().__init__(pruning_cfg)

    def forward(self):
        super().forward()

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
        return self.get_sampling_params()
    
class EdgeMaskIterativeSampler(MaskSampler):
    def __init__(self, pruning_cfg, reverse=True):
        super().__init__(pruning_cfg)
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
            prune_mask[f"attn-{self.component_type}"][self.cur_layer] = self.sample_prune_mask(
                attn_unif, 
                self.sampling_params[f"attn-{self.component_type}"][self.cur_layer]
            ).clone()

        mlp_unif = torch.rand((
            self.pruning_cfg.n_samples * self.pruning_cfg.batch_size, 
            *self.sampling_params[f"mlp-{self.component_type}"][self.cur_layer].shape[:-1]
        )).to(self.pruning_cfg.device)
        prune_mask[f"mlp-{self.component_type}"][self.cur_layer] = self.sample_prune_mask(
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

class ConstantMaskSampler():
    def __init__(self):
        self.sampled_mask = None

    def set_mask(self, mask):
        self.sampled_mask = mask

    def __call__(self):
        return None