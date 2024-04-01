
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

class MaskSampler(torch.nn.Module):
    def __init__(self, pruning_cfg, complexity_mean=False):
        super().__init__()

        self.complexity_mean = complexity_mean
        self.pruning_cfg = pruning_cfg
        self.log_columns = ['complexity_loss', 'temp', 'temp_cond', 'temp_count', 'temp_reg']

        self.sampling_params = torch.nn.ParameterDict({
            k: torch.nn.ParameterList([
                torch.nn.Parameter(p_init) for p_init in pruning_cfg.init_params[k]
            ]) for k in pruning_cfg.init_params
        })

        self.sampled_mask = None
        self.use_temperature = True
        self.temp_c = 0
        self.node_reg = 0
        self.def_value = 2/3
        self.sampling_function = self.sample_hard_concrete

        for param in self.parameters():
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))
        
    def get_sampling_params(self):
        # returns N x 2 tensor
        return torch.cat([ts.flatten(start_dim=0, end_dim=-2) if len(ts.shape) > 1 else ts.unsqueeze(0) for k in self.sampling_params for ts in self.sampling_params[k]], dim=0)
    
    def sample_hard_concrete(self, unif, sampling_params):
        # back prop against log alpha
        endpts = self.pruning_cfg.hard_concrete_endpoints
        concrete = (((.001+unif).log() - (1-unif).log() + sampling_params[...,0])/(sampling_params[...,1].relu()+.001)).sigmoid()

        hard_concrete = ((concrete + endpts[0]) * (endpts[1] - endpts[0])).clamp(0,1)

        # n_layers x (total_samples = batch_size * n_samples) x n_heads
        return hard_concrete
    
    def sample_mask(self):
        bsz = self.pruning_cfg.n_samples * self.pruning_cfg.batch_size
        prune_mask = {}
        for k in self.sampling_params:
            prune_mask[k] = []
            for i in range(len(self.sampling_params[k])):
                # if sampling_params[k][i].nelement() == 0:
                #     prune_mask[k].append(None)
                #     continue
                unif = torch.rand((bsz, *self.sampling_params[k][i].shape[:-1])).to(self.pruning_cfg.device)
                prune_mask[k].append(self.sampling_function(unif, self.sampling_params[k][i]))

        self.sampled_mask = prune_mask
        
    def fix_nans(self):
        fixed = True
        with torch.no_grad():
            sampling_params = self.get_sampling_params()
            
            nancount = sampling_params.isnan().sum()

            if nancount > 0:
                print("NANs", nancount)
                for k in self.sampling_params:
                    for ts in self.sampling_params[k]:
                        ts[ts[:,1].isnan().nonzero()[:,0],-1] = self.def_value
                        if ts.isnan().sum() > 0:
                            fixed = False
        return fixed
    
    def set_temp_c(self, temp_c):
        self.temp_c = temp_c

    # beta and alpha should be same shape as x, or broadcastable
    # def f_concrete(x, beta, alpha):
    #     return ((x.log() - (1-x).log()) * beta - alpha.log()).sigmoid()

    def complexity_loss(self, sampling_params):
        return (sampling_params[...,0]-sampling_params[...,1].relu() * (math.log(-self.pruning_cfg.hard_concrete_endpoints[0]/self.pruning_cfg.hard_concrete_endpoints[1]))).sigmoid()

    def get_mask_loss(self):
        all_sampling_params = self.get_sampling_params()

        # alphas already logged
        complexity_loss = self.complexity_loss(all_sampling_params)
                    
        temperature_loss = all_sampling_params[...,1].square()

        mask_loss = self.pruning_cfg.lamb * complexity_loss.sum() + self.temp_c * temperature_loss.sum()

        with torch.no_grad():
            avg_temp = all_sampling_params[...,1].relu().mean().item()
            temp_cond = torch.nan_to_num((all_sampling_params[...,1]-1).relu().sum() / (all_sampling_params[...,1] > 1).sum(), nan=0, posinf=0, neginf=0).item() + 1
            temp_count = (2*all_sampling_params[:,1].relu().sigmoid()-1).mean().item()

            print("Complexity:", complexity_loss.sum().item(), "out of", complexity_loss.nelement())
            print("Avg temperature", avg_temp)
            print("Avg temp > 1", temp_cond)
            print("Temp count", temp_count)

        mask_details = {                
            "complexity_loss": complexity_loss.mean().item() if self.complexity_mean else complexity_loss.sum().item(),
            "temp": avg_temp,
            "temp_cond": temp_cond,
            "temp_count": temp_count,
            "temp_reg": self.temp_c
        }
        return mask_loss, mask_details
    
    def forward(self):
        self.sample_mask()
        return self.get_mask_loss() 

    def take_snapshot(self, j):
        pass

    def load_snapshot(self):
        pass

    def record_state(self, j):
        all_sampling_params = self.get_sampling_params()

        sns.histplot(torch.cat([ts.flatten() for k in self.sampled_mask for ts in self.sampled_mask[k]], dim=0).detach().flatten().cpu())
        plt.savefig(f"{self.pruning_cfg.folder}/mask{j}.png")
        plt.close()

        sns.histplot(x=all_sampling_params[:,0].sigmoid().detach().flatten().cpu(), y=all_sampling_params[:,1].detach().flatten().cpu(), bins=100)
        plt.savefig(f"{self.pruning_cfg.folder}/params-probs{j}.png")
        plt.close()

        sns.histplot(x=all_sampling_params[:,0].detach().flatten().cpu(), y=all_sampling_params[:,1].detach().flatten().cpu(), bins=100)
        plt.savefig(f"{self.pruning_cfg.folder}/params-logits{j}.png")
        plt.close()

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
    
    # maximum scale = 2
    # more scale = more Unif
    def sample_bernoulli(self, unif, sampling_params, scale=1):
        probs = sampling_params[...,0].sigmoid()
        return (unif < probs.detach()) * 1
    
    # use sample estimate of mask loss instead of exact expectation to denoise gradient signal
    def sample_mask(self, constant=100, bottom_quantile=.75):
        bsz = self.pruning_cfg.n_samples * self.pruning_cfg.batch_size
        big_bsz = bsz * constant
        cand_mask = {}
        for k in self.sampling_params:
            cand_mask[k] = []
            for i in range(len(self.sampling_params[k])):
                # if sampling_params[k][i].nelement() == 0:
                #     prune_mask[k].append(None)
                #     continue
                unif = torch.rand((big_bsz, *self.sampling_params[k][i].shape[:-1])).to(self.pruning_cfg.device)
                cand_mask[k].append(self.sampling_function(unif, self.sampling_params[k][i]))

        filtered_mask = prune_dangling_edges(cand_mask, bsz=big_bsz)
        
        for k in cand_mask:
            for i, ts in enumerate(cand_mask[k]):
                cand_mask[k][i] = ts * filtered_mask[k][i].detach()
        
        with torch.no_grad():
            log_probs = self.compute_log_probs(cand_mask)

        cutoff = log_probs.quantile(bottom_quantile + (1-bottom_quantile-2/constant) * torch.rand(1).item())
        vals, indices = torch.topk(-1 * (log_probs > cutoff) * log_probs, bsz)
        # sns.histplot(vals.cpu())

        mask_loss = torch.zeros(bsz,).to(self.pruning_cfg.device)
        prune_mask = {}
        for k in self.sampling_params:
            prune_mask[k] = []
            for i in range(len(self.sampling_params[k])):
                # unif = torch.rand((bsz, *self.sampling_params[k][i].shape[:-1])).to(self.pruning_cfg.device)
                # prune_mask[k].append(self.sampling_function(unif, self.sampling_params[k][i]))
                prune_mask[k].append(cand_mask[k][i][indices])     
                mask_loss = mask_loss + prune_mask[k][i].flatten(start_dim=1,end_dim=-1).sum(dim=1)   
        self.sampled_mask = prune_mask
        self.mask_loss = mask_loss

    def compute_log_probs(self, prune_mask):
        log_probs = []
        for k in prune_mask:
            for i, ts in enumerate(prune_mask[k]):
                log_prob = self.sampling_params[k][i].squeeze(-1).sigmoid().log() * ts 
                + (1-self.sampling_params[k][i].squeeze(-1).sigmoid()).log() * (1-ts)
                log_probs.append(log_prob.flatten(start_dim=1,end_dim=-1).sum(dim=1))
        
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def get_mask_loss(self):
        all_sampling_params = self.get_sampling_params()

        # alphas already logged
        complexity_loss = self.complexity_loss(all_sampling_params)
        mask_loss = self.pruning_cfg.lamb * self.mask_loss

        with torch.no_grad():
            print("Complexity:", complexity_loss.sum().item(), "out of", complexity_loss.nelement())

        mask_details = {                
            "complexity_loss": complexity_loss.mean().item() if self.complexity_mean else complexity_loss.sum().item(),
        }
        return mask_loss, mask_details
        
    def forward(self):
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