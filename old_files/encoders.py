# %%
import torch as t
from fancy_einsum import einsum
import math

# %%
class UntiedEncoder(t.nn.Module):
    def __init__(self, feature_dim, activation_dim):
        super().__init__()
        self.feature_weights = t.nn.Parameter(t.normal(0, 1/math.sqrt(feature_dim), (feature_dim, activation_dim)))
        self.encoder_weights = t.nn.Parameter(t.normal(0, 1/math.sqrt(feature_dim), (feature_dim, activation_dim)))
        self.bias = t.nn.Parameter(t.rand(feature_dim))
        self.relu = t.nn.ReLU()
        self.floating_mean = t.nn.Parameter(t.rand(activation_dim))
        
    def forward(self,x):
        features = (einsum("features orig, batch orig -> batch features", self.encoder_weights, x - self.floating_mean) + self.bias).relu()
        l1 = features.sum()
        l2 = features.square().sum()
        recovered = einsum("features recovered, batch features -> batch recovered", self.feature_weights, features)
        # print(self.feature_weights.isnan().sum())
        # print(self.feature_weights.shape)
        # print(recovered.shape)
        return recovered + self.floating_mean, l1, l2

# %%

class TiedEncoder(t.nn.Module):
    def __init__(self, feature_dim, activation_dim):
        super().__init__()
        self.feature_weights = t.nn.Parameter(t.normal(0, 1/math.sqrt(feature_dim), (feature_dim, activation_dim)))
        # self.encoder_weights = t.nn.Parameter(t.normal(0, 1/math.sqrt(feature_dim), (feature_dim, activation_dim)))
        self.bias = t.nn.Parameter(t.rand(feature_dim))
        self.relu = t.nn.ReLU()
        self.floating_mean = t.nn.Parameter(t.rand(activation_dim))
        
    def forward(self,x):
        features = (einsum("features orig, batch orig -> batch features", self.feature_weights, x - self.floating_mean) + self.bias).relu()
        l1 = features.sum()
        l2 = features.square().sum()
        recovered = einsum("features recovered, batch features -> batch recovered", self.feature_weights, features)
        return recovered + self.floating_mean, l1, l2

