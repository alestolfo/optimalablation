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