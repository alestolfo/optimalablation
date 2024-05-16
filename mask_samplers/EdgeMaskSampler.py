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
        self.temp_scale = 1
        self.log_columns = ['complexity_loss']
        self.use_temperature = False
    
    # maximum scale = 2
    # more scale = more Unif
    def sample_modified_unif(self, unif, sampling_params):
        probs = sampling_params[...,0].sigmoid()
        window_sz = (self.temp_scale * probs * (1 - probs)).detach()

        # we want scale invariance
        probs = window_sz * probs - (window_sz - 1) * probs.detach()
        return 1-((unif - probs) / window_sz + 0.5).clamp(0,1)

    def sample_bernoulli(self, unif, sampling_params):
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

        sns.histplot(torch.cat([ts.flatten() for k in self.sampled_mask for ts in self.sampled_mask[k]], dim=0).detach().flatten().cpu())
        plt.savefig(f"{self.pruning_cfg.folder}/mask{j}.png")
        plt.close()

        sns.histplot(x=all_sampling_params.sigmoid().detach().flatten().cpu(), bins=100)
        plt.savefig(f"{self.pruning_cfg.folder}/params-probs{j}.png")
        plt.close()
    
    def clip_grad(self, bound):
        grad_norms = []
        for k in self.sampling_params:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.sampling_params[k], max_norm=float('inf'))
            grad_norms.append(grad_norm.item())
            torch.nn.utils.clip_grad_norm_(self.sampling_params[k], max_norm=bound)
        return grad_norms

class EdgeMaskUnifWindowSampler(EdgeMaskUnifSampler):
    def __init__(self, pruning_cfg, node_reg=0, default_window=0.25):
        super().__init__(pruning_cfg, node_reg)

        self.sampling_function = self.sample_modified_unif
        self.def_value = 0
        self.temp_scale = 1
        self.log_columns = ['complexity_loss']
        self.use_temperature = False

        self.running_mean = torch.zeros_like(self.get_sampling_params()).to(pruning_cfg.device)
        self.running_variance = torch.zeros_like(self.running_mean).to(pruning_cfg.device)
        self.running_batches = torch.zeros_like(self.running_mean).to(pruning_cfg.device)
        self.running_samples = torch.zeros_like(self.running_mean).to(pruning_cfg.device)

        self.edge_count = self.running_mean.shape[0]
        self.selected_edges = None
        self.selected_edge_mask = None
        self.selected_edge_unif = None
        self.default_window = default_window
        self.windows = torch.ones_like(self.running_mean).to(pruning_cfg.device) * default_window
    
    def sample_mask_single_edge(self):
        bsz = self.pruning_cfg.n_samples * self.pruning_cfg.batch_size
        device = self.pruning_cfg.device
        
        self.selected_edges = torch.randperm(self.edge_count)[:bsz].to(device)
        selected_edge_mask = torch.zeros((bsz, self.edge_count)).to(device)
        selected_edge_mask[torch.arange(selected_edge_mask.shape[0]).to(device), self.selected_edges] = 1
        current_edges = 0

        self.selected_edge_mask = {}
        self.selected_edge_unif = {}
        prune_mask = {}
        for k in self.sampling_params:
            prune_mask[k] = []
            self.selected_edge_mask[k] = []
            self.selected_edge_unif[k] = []

            for ts in self.sampling_params[k]:
                section_selected_mask = selected_edge_mask[:,current_edges:current_edges + ts.nelement()].reshape((bsz, *ts.shape[:-1]))
                unif = torch.rand((bsz, *ts.shape[:-1])).to(device)

                self.selected_edge_mask[k].append(section_selected_mask)
                self.selected_edge_unif[k].append(unif)

                section_mask = (1-section_selected_mask) * self.sampling_function(unif, ts).detach() + section_selected_mask * self.sample_definite_unif(unif, ts)

                prune_mask[k].append(section_mask)

                current_edges += ts.nelement()
        self.sampled_mask = prune_mask
    
    def sample_mask_only_edge(self):
        if self.selected_edge_mask is None or self.selected_edge_unif is None:
            raise Exception("No selected edge mask")

        prune_mask = {}
        for k in self.sampling_params:
            prune_mask[k] = []
            for i, ts in enumerate(self.sampling_params[k]):
                section_selected_mask = self.selected_edge_mask[k][i]
                section_unif = self.selected_edge_unif[k][i]

                section_mask = (1-section_selected_mask) * self.sampling_function(section_unif, ts).detach() + section_selected_mask * self.sample_definite_unif(section_unif, ts)

                requires_settle = (section_mask > 0) * (section_mask < 1) * (1 - section_selected_mask)
                unif = torch.rand(section_mask.shape).to(self.pruning_cfg.device)
                settled_outcome = self.sample_bernoulli(unif, ts)

                prune_mask[k].append(requires_settle * settled_outcome + (1-requires_settle) * section_mask)
            
        self.sampled_mask = prune_mask
    # maximum scale = 2
    # more scale = more Unif
    # Perform PIT on Unif to map to mixture of Bernoulli and Unif
    def sample_modified_unif(self, unif, sampling_params):
        probs = sampling_params[...,0].sigmoid()
        return 1-((unif - probs) / (self.temp_scale * probs * (1 - probs)).detach() + 0.5).clamp(0,1)

    def sample_window_mask(self):
        bsz = self.pruning_cfg.n_samples * self.pruning_cfg.batch_size
        current_edges = 0

        prune_mask = {}
        for k in self.sampling_params:
            prune_mask[k] = []
            for i, ts in enumerate(self.sampling_params[k]):
                # if ts.nelement() == 0:
                #     prune_mask[k].append(None)
                #     continue
                windows = self.windows[current_edges:current_edges + ts.nelement()].reshape(ts.shape[:-1])
                unif = torch.rand((bsz, *ts.shape[:-1])).to(self.pruning_cfg.device)
                prune_mask[k].append(self.sample_window_unif(unif, ts, windows))
                # prune_mask[k].append(self.sample_modified_unif(unif, ts))
                current_edges += ts.nelement()
        self.sampled_mask = prune_mask

    def sample_window_unif(self, unif, sampling_params, windows):
        probs = sampling_params[...,0].sigmoid()
        windows = torch.minimum(torch.minimum(2 * probs, 2 * (1-probs)), windows)
        return 1-(((unif - probs) / windows.detach()) + 0.5).clamp(0,1)

    # sample Unif, but gradient wrt. params is as if it were the mixture
    def sample_definite_unif(self, unif, sampling_params):
        probs = sampling_params[...,0].sigmoid()
        return unif + (probs - probs.detach()) / (self.temp_scale * probs * (1 - probs)).detach()
    
    def forward(self):
        return super().forward()
    
    def get_sampling_grads(self):
        return torch.cat([
            ts.grad.flatten(start_dim=0, end_dim=-2) 
            if len(ts.shape) > 1 
            else ts.grad.unsqueeze(0) 
            for k in self.sampling_params 
            for ts in self.sampling_params[k]
        ], dim=0)
    
    def compile_grads(self):
        # N x 1
        grads = self.get_sampling_grads()

        with torch.no_grad():
            # N x 1
            new_samples = torch.cat([
                ((ts > 0) * (ts < 1)).sum(dim=0).flatten()
                for k in self.sampled_mask 
                for ts in self.sampled_mask[k]
            ], dim=0).unsqueeze(-1)

            grads = torch.where(
                new_samples > 0,
                grads,
                0
            )
            self.running_mean, self.running_variance, self.running_batches, self.running_samples = update_means_variances_mixed(self.running_mean, self.running_variance, grads, self.running_batches, self.running_samples, new_samples)

    def compile_grads_exponential(self):
        # N x 1
        grads = self.get_sampling_grads()

        with torch.no_grad():
            # N x 1
            new_samples = torch.cat([
                ((ts > 0) * (ts < 1)).sum(dim=0).flatten()
                for k in self.sampled_mask 
                for ts in self.sampled_mask[k]
            ], dim=0).unsqueeze(-1)

            grads = torch.where(
                new_samples > 0,
                grads,
                0
            )
            self.running_mean, self.running_variance, self.running_batches, self.running_samples = update_means_variances_exponential(self.running_mean, self.running_variance, grads, self.running_batches, self.running_samples, new_samples)

    def update_window(self, new_default, window_min=5e-3):
        with torch.no_grad():
            sampling_params = self.get_sampling_params()
            params_deriv = sampling_params.sigmoid() * (1 - sampling_params.sigmoid())

            relative_weights = (self.running_variance + 1e-8).sqrt() * params_deriv
            self.windows = (self.default_window * relative_weights / relative_weights.mean()).clamp(min=window_min, max=1)

            self.default_window = new_default
    
    def reset_grads(self):
        self.running_mean = torch.zeros_like(self.get_sampling_params()).to(self.pruning_cfg.device)
        self.running_variance = torch.zeros_like(self.running_mean).to(self.pruning_cfg.device)
        self.running_batches = torch.zeros_like(self.running_mean).to(self.pruning_cfg.device)
        self.running_samples = torch.zeros_like(self.running_mean).to(self.pruning_cfg.device)

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