# %%

import glob
import os
import shutil

for ds in ["ioi", "gt"]:
    for ablation in ["cf", "mean", "resample", "oa"]:
        for tech in ["acdc", "eap", "hc", "unif"]:
            f_path = f"results/pruning/{ds}/{ablation}/{tech}"
            cp_path = f"reqfiles/circ/{ds}/{ablation}/{tech}"
            if not os.path.exists(cp_path):
                os.makedirs(cp_path)
            shutil.copy(f"{f_path}/post_training.pkl", f"{cp_path}/pt.pkl")
                
                

# %%
import torch
import numpy as np
# %%
all_loss = []
g = glob.glob("results/pruning_random/ioi/cf/*")
for f in g:
    my_losses = torch.load(f)
    all_loss += (my_losses['loss'])
# %%
# %%
print(np.mean(all_loss))
print(np.std(all_loss))
# %%
import pickle

# %%
import io
from utils.training_utils import LinePlot

# %%
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open("results/pruning/ioi/cf/unif/0.0005/metadata.pth", "rb") as f:
    contents = CPU_Unpickler(f).load()

# %%
torch.load("results/pruning/ioi/cf/unif/post_training.pkl", map_location=torch.device('cpu'))# %%

# %%
