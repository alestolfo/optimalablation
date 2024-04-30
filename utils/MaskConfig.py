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
import glob
import torch.optim
import os
import time
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.circuit_utils import edge_prune_mask, vertex_prune_mask, retrieve_mask, discretize_mask, prune_dangling_edges, get_ioi_nodes, mask_to_edges, nodes_to_mask, nodes_to_vertex_mask, mask_to_nodes, edges_to_mask, get_ioi_edge_mask
from utils.training_utils import LinePlot

# %%

class InferenceConfig:
    def __init__(self, device, folder, cfg):
        self.device = device
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads

        self.folder = folder
        self.lamb = None
        self.record_every = 100
        self.checkpoint_every = 5

        # as in the louizos paper
        self.starting_beta = 2/3
        self.hard_concrete_endpoints = (-0.1, 1.1)

        self.layers_to_prune = []
        for layer_no in range(self.n_layers):
            self.layers_to_prune.append(("attn", layer_no))
            self.layers_to_prune.append(("mlp", layer_no))
        self.layers_to_prune.append(("mlp", self.n_layers))

        self.temp_min_reg = 0.001
        self.temp_adj_intv = 10
        self.temp_avg_intv = 20
        self.temp_comp_intv = 200
        self.temp_convergence_target = 2000
        self.temp_c = 0

        self.temp_momentum = 0
    
    def initialize_params(self, init_param, init_scale):
        if init_scale is None:
            self.init_params = {
                k: [
                    torch.stack(
                        [mask_tensor.squeeze(0).to(self.device) * init_param, 
                        torch.ones(mask_tensor.shape[1:]).to(self.device) * self.starting_beta], 
                    dim=-1)
                    for mask_tensor in self.constant_prune_mask[k]
                ]
                for k in self.constant_prune_mask
            }
        else:
            self.init_params = {
                k: [
                    torch.stack(
                        [torch.randn(mask_tensor.shape[1:]) * init_scale + init_param, 
                        torch.ones(mask_tensor.shape[1:]) * self.starting_beta], 
                    dim=-1).to(self.device)
                    for mask_tensor in self.constant_prune_mask[k]
                ]
                for k in self.constant_prune_mask
            }

    def initialize_params_probs(self, init_param):
        self.init_params = {
            k: [
                mask_tensor.squeeze(0).to(self.device).unsqueeze(-1) * init_param
                for mask_tensor in self.constant_prune_mask[k]
            ]
            for k in self.constant_prune_mask
        }

    def reset_temp(self):
        self.temp_c = 0

    def temp_scheduler(self, log):
        avg_intv = self.temp_avg_intv
        comp_intv = self.temp_comp_intv
        if log.t % self.temp_adj_intv != 0:
            return self.temp_c
        
        # how many % did the temperature loss decline
        g = log.stat_sig_growth("temp_cond", avg_intv, comp_intv)
        if log.t - log.last_tick < max(20, 1.5 * comp_intv) or g == False:
            return 0            
        cur_temp = np.mean(log.stat_book["temp_cond"][-20:])
        decline_rate, growth_rate = g
        time_left = max(self.temp_convergence_target - (log.t - log.last_tick), 50)
        target_decline_rate = 1-np.exp(
            (-1 * np.log(cur_temp)) / (time_left / comp_intv))
        
        self.temp_c = max(self.temp_c, self.temp_min_reg)

        if growth_rate > 0:
            self.temp_momentum += 1
            if self.temp_momentum * self.temp_adj_intv > time_left // 10:
                self.temp_c *= 1.05
        elif cur_temp < 1.01:
            self.temp_c *= 0.95
        else:
            self.temp_momentum = max(0, self.temp_momentum - 1)
            if self.temp_momentum > 0:
                self.temp_c *= 0.95
            elif decline_rate <= 0.001:
                self.temp_c *= 1.05
            elif decline_rate > target_decline_rate:
                self.temp_momentum = 0
                self.temp_c *= max(1 + (target_decline_rate / decline_rate - 1) / 3, 0.5)            
            else:
                self.temp_c *= min(1 + (target_decline_rate / decline_rate - 1) / 10, 1.5)   
        return self.temp_c

    def take_snapshot(self, component_pruner, lp_count, sampling_optimizer, modal_optimizer, j):
        snapshot_path = f"{self.folder}/snapshot{j}.pth"
        metadata_path = f"{self.folder}/metadata{j}.pkl"

        lp_count.plot(save=f"{self.folder}/train-step{j}.png")

        plot_1 = ['kl_loss', 'complexity_loss']
        if 'node_loss' in component_pruner.log.stat_book:
            plot_1.append('node_loss')
        component_pruner.log.plot(plot_1, save=f"{self.folder}/train-loss{j}.png")

        if 'temp' in component_pruner.log.stat_book:
            component_pruner.log.plot(['temp', 'temp_cond', 'temp_count', 'temp_reg'], save=f"{self.folder}/train-temp{j}.png")

        pruner_dict = component_pruner.state_dict()
        pruner_dict = {k: pruner_dict[k] for k in pruner_dict if not k.startswith("base_model")}
        torch.save({
            'pruner_dict': pruner_dict,
            'sampling_optim_dict': sampling_optimizer.state_dict(),
            'modal_optim_dict': modal_optimizer.state_dict(),
        }, snapshot_path)

        with open(metadata_path, "wb") as f:
            pickle.dump((component_pruner.log, lp_count, (self.temp_c, self.temp_momentum)), f)
        
        component_pruner.mask_sampler.take_snapshot(j)

    def load_snapshot(self, component_pruner, sampling_optimizer, modal_optimizer, gpu_requeue=False, pretrained_folder=None):
        snapshot_path = f"{self.folder}/snapshot.pth"
        metadata_path = f"{self.folder}/metadata.pkl"

        if gpu_requeue and os.path.exists(snapshot_path) and os.path.exists(metadata_path):
            print("Loading previous training run")
            previous_state = torch.load(snapshot_path)
            component_pruner.load_state_dict(previous_state['pruner_dict'], strict=False)
            sampling_optimizer.load_state_dict(previous_state['sampling_optim_dict'])
            modal_optimizer.load_state_dict(previous_state['modal_optim_dict'])

            with open(metadata_path, "rb") as f:
                x = pickle.load(f)
                main_log = x[0]
                lp_count = x[1]
                if len(x) == 3:
                    self.temp_c = x[2][0]
                    self.temp_momentum = x[2][1]
                
            component_pruner.set_log(main_log)
            component_pruner.mask_sampler.load_snapshot()
            return lp_count
        
        if pretrained_folder is not None:
            pretrained_snapshot_path = f"{pretrained_folder}/snapshot.pth"
            print("Loading pretrained weights")
            previous_state = torch.load(pretrained_snapshot_path)
            component_pruner.load_state_dict(previous_state['pruner_dict'], strict=False)
        
        return LinePlot(['step_size', 'mode_step_size', 'max_grad_norm'])
    
    def record_post_training(self, mask_sampler, component_pruner, ds_test, next_batch, in_format="edges", out_format="edges", load_edges=False):
        log = {"lamb": [], "tau": [], "losses": []}
        if out_format == "edges":
            log["edges"] = []
            log["clipped_edges"] = []
            log["vertices"] = []
        
        for lamb_path in glob.glob(f"{self.folder}/*"):
            lamb = lamb_path.split("/")[-1]
            print(lamb)
            # if lamb =="manual":
                # if in_format == "edges":
                #     prune_mask = get_ioi_edge_mask()
                #     # prune_mask = edges_to_mask(ioi_edges)
                # else:
                #     ioi_nodes = get_ioi_nodes()
                #     prune_mask = nodes_to_vertex_mask(ioi_nodes)
            if load_edges and (lamb == "manual" or lamb[1] == "." or lamb[1:3] == "e-"):
                edge_list = torch.load(f"{self.folder}/edges_{lamb}.pth")
                prune_mask = edges_to_mask(edge_list)
            else:
                try:
                    float(lamb[-1])
                    float(lamb[0])
                except:
                    continue
                prune_mask = retrieve_mask(lamb_path)

            files = glob.glob(f"{lamb_path}/fit_modes_*.pth")
            for tau_path in files:
                tau = tau_path.split("/")[-1].replace("fit_modes_", "").replace(".pth", "")
                print(tau)
                try:
                    tau = float(tau)
                except: 
                    continue
                
                discrete_mask = discretize_mask(prune_mask, tau)
                                    
                if in_format=="edges":
                    discrete_mask, edges, clipped_edges, attn_vertices, mlp_vertices = prune_dangling_edges(discrete_mask)
                    total_vertices = (attn_vertices > 0).sum() + (mlp_vertices > 0).sum()
                elif out_format=="edges":
                    vertices, total_vertices = mask_to_nodes(discrete_mask, mask_type="nodes")
                    _, edges = mask_to_edges(nodes_to_mask(vertices, all_mlps=False))
                    clipped_edges = edges

                mask_sampler.set_mask(discrete_mask)

                component_pruner.load_state_dict(torch.load(tau_path), strict=False)

                ds_iter = iter(ds_test)
                kl_losses = []

                for i in tqdm(range(20)):
                    batch, last_token_pos = next_batch(next(ds_iter))
                    with torch.no_grad():
                        loss = component_pruner(batch, last_token_pos, timing=False)
                    kl_losses.append(loss.mean().item())
                avg_loss = np.mean(kl_losses)
                log["lamb"].append(lamb)
                log["tau"].append(tau)
                log["edges"].append(edges)
                log["clipped_edges"].append(clipped_edges)
                log["vertices"].append(total_vertices)
                log["losses"].append(avg_loss)
                print("Clipped edges", clipped_edges)
                print("Avg KL loss", avg_loss)

        with open(f"{self.folder}/post_training.pkl", "wb") as f:
            pickle.dump(log, f)


# %%
    
class EdgeInferenceConfig(InferenceConfig):
    def __init__(self, cfg, device, folder, batch_size=None, init_param=-0.5, init_scale=None, prior=None, prior_scale=0):
        super().__init__(device, folder, cfg)

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 5

        self.n_samples = 12
        self.constant_prune_mask = edge_prune_mask

        self.lr = 1e-1
        self.lr_modes = 1e-3

        self.initialize_params(init_param, init_scale)

class VertexInferenceConfig(InferenceConfig):
    def __init__(self, cfg, device, folder, batch_size=None, init_param=-0.5, init_scale=None):
        super().__init__(device, folder, cfg)
        
        if batch_size is None:
            self.batch_size = 10
        else:
            self.batch_size = batch_size

        self.n_samples = 25
        self.constant_prune_mask = vertex_prune_mask

        self.lr = 1e-2
        self.lr_modes = 1e-3

        self.initialize_params(init_param, init_scale)