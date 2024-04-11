# %%
import torch
from sys import argv
from functools import partial
import torch.optim
from VertexPruner import VertexPruner
from MaskSampler import ConstantMaskSampler
from MaskConfig import VertexInferenceConfig
from task_datasets import IOIConfig, GTConfig
from training_utils import load_model_data, LinePlot

# %%
# load model
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
# model.cfg.use_attn_result = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# settings
try:
    reg_lamb = float(argv[1])
except:
    reg_lamb=1e-4

folder=f"pruning_vertices_auto/ioi_with_mlp"

batch_size=50
pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, batch_size=batch_size)
pruning_cfg.lamb = reg_lamb
pruning_cfg.n_samples = 1

task_ds = IOIConfig(batch_size, device)
ds_test = task_ds.get_test_set(tokenizer)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = ConstantMaskSampler()
vertex_pruner = VertexPruner(model, pruning_cfg, task_ds.init_modes(), mask_sampler, inference_mode=True)
vertex_pruner.add_patching_hooks()

# %%
next_batch = partial(task_ds.next_batch, tokenizer)
pruning_cfg.record_post_training(mask_sampler, vertex_pruner, ds_test, next_batch, in_format="nodes")

# %%
