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