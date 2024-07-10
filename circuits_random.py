# %%
# import torch
import os
from sys import argv
from tqdm import tqdm
import torch.optim
import pickle
from pruners.EdgePruner import EdgePruner
from mask_samplers.MaskSampler import ConstantMaskSampler
from utils.MaskConfig import EdgeInferenceConfig
from utils.task_datasets import get_task_ds
from utils.circuit_utils import discretize_mask, prune_dangling_edges
from utils.training_utils import load_model_data, LinePlot, load_args   
import seaborn as sns
import random
from functools import partial 
sns.set()

# %%
# load model
model_name = "gpt2-small"
owt_batch_size = 10
device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
model.eval()
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# %%
args = load_args("pruning_random", None, {"desc": "mean", "minwindow": 400, "maxwindow": 500, "tau": 200})
uid = random.randint(1,10000000)
print(uid)
folder, dataset, ablation_type, min_edges, max_edges, n_circuits = args["folder"], args["dataset"], args["desc"], args["minwindow"], args["maxwindow"], args["tau"]

folder = f"{folder}/{ablation_type}"
if not os.path.exists(folder):
    os.makedirs(folder)

if min_edges is None:
    min_edges = 400
if max_edges is None:
    max_edges = 500
if n_circuits is None:
    n_circuits = 30

print("Folder", folder)
print("Dataset", dataset)
print("Ablation type", ablation_type)
print("Min edges", min_edges)
print("Max edges", max_edges)
print("N circuits", n_circuits)

gpu_requeue = True
# reset_optim = 1000

batch_size = 75
pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, batch_size=batch_size)
pruning_cfg.n_samples = 1

task_ds = get_task_ds(dataset, batch_size, device, ablation_type)

for param in model.parameters():
    param.requires_grad = False

# %%
circuits = []
t = 0
edge_target = max_edges + 100
total_edges = sum([ts.nelement() for k in pruning_cfg.init_params for ts in pruning_cfg.init_params[k]])
rolling_edges = []
rolling_targets = []
while len(circuits) < n_circuits and t < n_circuits * 10:
    prob = edge_target / total_edges

    circuit_mask = {}
    circuit_edges = 0
    for k in pruning_cfg.init_params:
        circuit_mask[k] = []
        for ts in pruning_cfg.init_params[k]:
            rand = (torch.rand_like(ts.squeeze(-1).unsqueeze(0)).to(device) < prob) * 1
            circuit_edges += rand.sum().item()
            circuit_mask[k].append(rand)
    
    discrete_mask = discretize_mask(circuit_mask, 0.5)
    pruned_mask, e, c_e, _,_ = prune_dangling_edges(discrete_mask)
    rolling_edges.append(c_e)
    rolling_targets.append(edge_target)
    if c_e >= min_edges and c_e <= max_edges:
        circuits.append((pruned_mask, c_e))
    
    avg_edges = sum(rolling_edges[-5:]) / 5
    if avg_edges < min_edges:
        edge_target += 20
    elif avg_edges > max_edges:
        edge_target -= 20

# %%

sns.lineplot(rolling_edges)
sns.lineplot(rolling_targets)

# %%
mask_sampler = ConstantMaskSampler()

pruner_args = task_ds.get_pruner_args()
edge_pruner = EdgePruner(model, pruning_cfg, mask_sampler, **pruner_args)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

# %%
next_batch = partial(task_ds.retrieve_batch_cf, tokenizer)

all_losses = []
all_edge_counts = []
for i, (circuit, edge_count) in enumerate(circuits):
    mask_sampler.set_mask(circuit)

    if ablation_type == "oa":
        edge_pruner.reset_parameters(pruner_args['init_modes'])        
        modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=5 * pruning_cfg.lr_modes, weight_decay=0)

        max_batches = 200
        for no_batches in tqdm(range(max_batches)):

            modal_optimizer.zero_grad()
            batch, last_token_pos, cf = next_batch()
            loss = edge_pruner(batch, last_token_pos, cf, timing=False)
            loss.backward()
            modal_optimizer.step()
        
        edge_pruner.log.plot(start=1)
    
    no_test_batches = 20
    test_loss = []
    for no_batches in tqdm(range(no_test_batches)):
        batch, last_token_pos, cf = next_batch()
        with torch.no_grad():
            loss = edge_pruner(batch, last_token_pos, cf, timing=False)
            test_loss.append(loss.item())
    
    all_losses.append(sum(test_loss) / len(test_loss))
    all_edge_counts.append(edge_count)

    if i % -10 == -1:
        torch.save({"loss": all_losses, "edges": all_edge_counts}, f"{folder}/log_{uid}.pth")
# %%
