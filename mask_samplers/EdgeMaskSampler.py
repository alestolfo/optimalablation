
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
from ..circuit_utils import prune_dangling_edges, discretize_mask

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
        
        return F.tanh(2 * (node_losses_out + torch.stack(node_losses_in, dim=0)) / (n_layers * n_heads))        

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
        self.log_columns = ['complexity_loss']
        self.use_temperature = False
    
    # maximum scale = 2
    # more scale = more Unif
    def sample_modified_unif(self, unif, sampling_params, scale=1):
        probs = sampling_params[...,0].sigmoid()
        return 1-((unif - probs) / (scale * probs * (1 - probs)) + 0.5).clamp(0,1)

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

        sns.histplot(torch.cat([ts.flatten() for k in self.sampled_mask for ts in self.sampled_mask[k]], dim=0).detach().flatten().cpu())
        plt.savefig(f"{self.pruning_cfg.folder}/mask{j}.png")
        plt.close()

        sns.histplot(x=all_sampling_params.sigmoid().detach().flatten().cpu(), bins=100)
        plt.savefig(f"{self.pruning_cfg.folder}/params-probs{j}.png")
        plt.close()

class EdgeMaskBernoulliSampler(EdgeMaskUnifSampler):
    def __init__(self, pruning_cfg, node_reg=0):
        super().__init__(pruning_cfg, node_reg)

        self.sampling_function = self.sample_bernoulli
        self.fix_mask_prop = None    
        self.fixed_mask = None
    
    # maximum scale = 2
    # more scale = more Unif
    def sample_bernoulli(self, unif, sampling_params, scale=1):
        with torch.no_grad():
            probs = sampling_params[...,0].sigmoid()
            return (unif < probs.detach()) * 1
    
    # use sample estimate of mask loss instead of exact expectation to denoise gradient signal
    def sample_mask(self, constant=1, bottom_quantile=.75):
        bsz = self.pruning_cfg.n_samples * self.pruning_cfg.batch_size
        big_bsz = bsz * constant
        
        cand_mask = {}
        for k in self.sampling_params:
            cand_mask[k] = []
            for ts in self.sampling_params[k]:
                # if sampling_params[k][i].nelement() == 0:
                #     prune_mask[k].append(None)
                #     continue
                unif = torch.rand((big_bsz, *ts.shape[:-1])).to(self.pruning_cfg.device)
                cand_mask[k].append(self.sampling_function(unif, ts))

        if self.fix_mask_prop is not None:
            self.fixed_mask = {}
            # self.sampling_probs = {}
            for k in self.sampling_params:
                self.fixed_mask[k] = []
                # self.sampling_probs[k] = []
                for i,ts in enumerate(self.sampling_params[k]):
                    unif = torch.rand((1, *ts.shape[:-1])).to(self.pruning_cfg.device)
                    fixed_mask_bernoulli = self.sampling_function(unif, ts)

                    mixture_unif = torch.rand(ts.shape[:-1]).to(self.pruning_cfg.device)

                    cand_mask[k][i] = (
                        (mixture_unif <= self.fix_mask_prop) * fixed_mask_bernoulli
                        + (mixture_unif > self.fix_mask_prop) * cand_mask[k][i]
                    )

                    # self.sampling_probs[k].append(
                    #     (mixture_unif <= self.fix_mask_prop).unsqueeze(-1) * self.sampling_params[k][i].detach()
                    #     + (mixture_unif > self.fix_mask_prop).unsqueeze(-1) * self.sampling_params[k][i]
                    # )

                    self.fixed_mask[k].append((mixture_unif <= self.fix_mask_prop) * 1)
        
        self.sampled_mask = cand_mask
        # filtered_mask = prune_dangling_edges(cand_mask, bsz=big_bsz)
        
        # for k in cand_mask:
        #     for i, ts in enumerate(cand_mask[k]):
        #         cand_mask[k][i] = ts * filtered_mask[k][i].detach()
        
        # with torch.no_grad():
        #     log_probs = self.compute_log_probs(cand_mask)

        # cutoff = log_probs.quantile(bottom_quantile + (1-bottom_quantile-2/constant) * torch.rand(1).item())
        # vals, indices = torch.topk(-1 * (log_probs > cutoff) * log_probs, bsz)
        # # sns.histplot(vals.cpu())

        # mask_loss = torch.zeros(bsz,).to(self.pruning_cfg.device)
        # prune_mask = {}
        # for k in cand_mask:
        #     prune_mask[k] = []
        #     for ts in cand_mask[k]:
        #         # unif = torch.rand((bsz, *self.sampling_params[k][i].shape[:-1])).to(self.pruning_cfg.device)
        #         # prune_mask[k].append(self.sampling_function(unif, self.sampling_params[k][i]))
        #         ts = ts[indices]
        #         prune_mask[k].append(ts)     
        #         mask_loss = mask_loss + ts.flatten(start_dim=1,end_dim=-1).sum(dim=1)   

        # self.sampled_mask = prune_mask
        # self.mask_loss = mask_loss

    def compute_log_probs(self, prune_mask):
        if self.fix_mask_prop is None:
            sampling_probs = self.sampling_params
        else:
            # don't perform gradient updates on fixed mask items
            sampling_probs = {}
            for k in self.sampling_params:
                sampling_probs[k] = []
                for i,ts in enumerate(self.sampling_params[k]):
                    sampling_probs[k].append(
                        self.fixed_mask[k][i].unsqueeze(-1) * ts.detach() 
                        + (1-self.fixed_mask[k][i].unsqueeze(-1)) * ts
                    )
        # sampling_probs = self.sampling_params

        log_probs = []
        for k in prune_mask:
            for i, ts in enumerate(prune_mask[k]):
                log_prob = (
                    sampling_probs[k][i].squeeze(-1).sigmoid().log() * ts 
                    + (1-sampling_probs[k][i].squeeze(-1).sigmoid()).log() * (1-ts)
                )
                log_probs.append(log_prob.flatten(start_dim=1,end_dim=-1).sum(dim=1))
        
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def get_mask_loss(self):
        all_sampling_params = torch.cat([
            (ts * (1-self.fixed_mask[k][i]).unsqueeze(-1)).flatten(start_dim=0,end_dim=-2)
            for k in self.sampling_params 
            for i, ts in enumerate(self.sampling_params[k])
        ], dim=0)

        # alphas already logged
        complexity_loss = self.complexity_loss(all_sampling_params)
        # mask_loss = self.pruning_cfg.lamb * self.mask_loss

        with torch.no_grad():
            print("Complexity:", complexity_loss.sum().item(), "out of", complexity_loss.nelement())

        mask_details = {                
            "complexity_loss": complexity_loss.mean().item() if self.complexity_mean else complexity_loss.sum().item(),
        }
        return complexity_loss, mask_details
        
    def forward(self):
        if not self.fix_mask:
            self.sample_mask()
        
        mask_loss, mask_details = self.get_mask_loss()

        if self.node_reg > 0:
            node_complexity = self.node_reg_loss().sum()
            node_reg_loss = self.node_reg * node_complexity
            mask_loss = mask_loss + node_reg_loss
            mask_details["node_loss"] = node_complexity.item()
            print("Node complexity:", node_complexity.item())

        return mask_loss, mask_details

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

class ConstantMaskSampler():
    def __init__(self):
        self.sampled_mask = None
        self.use_temperature = False
        self.log_columns = []

    def set_mask(self, mask):
        self.sampled_mask = mask

    def __call__(self):
        return 0, {}

    def record_state(self, j):
        pass

# for attribution patching
class AttributionPatchingMaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg):
        super().__init__()

        self.use_temperature = False
        self.log_columns = []

        n_layers = pruning_cfg.n_layers
        n_heads = pruning_cfg.n_heads

        self.device = pruning_cfg.device
        self.bsz = pruning_cfg.batch_size

        self.sampling_params = torch.nn.ParameterDict({
            "attn": torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((n_heads,)).to(self.device)) 
                for _ in range(n_layers)
            ]),
            "mlp": torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(()).to(self.device)) 
                for _ in range(n_layers)
            ])
        })

    # even though the same mask is taken every time, we need to recompute it to take derivatives
    def forward(self):
        self.sampled_mask = {}
        for k in self.sampling_params:
            self.sampled_mask[k] = []
            for ts in self.sampling_params[k]:
                self.sampled_mask[k].append(torch.ones((self.bsz, *ts.shape)).to(self.device) * ts)

        return 0, {}

    def record_state(self, j):
        pass

# for direct mean ablation. Can't take derivatives
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
                torch.nn.Parameter(torch.zeros(()).to(self.device)) 
                for i in range(self.n_layers)
            ])
        })

    def forward(self):
        bsz = self.pruning_cfg.batch_size * self.pruning_cfg.n_samples

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
                torch.ones((bsz,)).to(self.device) 
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
