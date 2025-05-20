# %%
import torch
import datasets
import os
from sys import argv
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
import argparse
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from mask_samplers.EdgeMaskSampler import EdgeMaskUnifSampler
from utils.MaskConfig import EdgeInferenceConfig
from utils.task_datasets import get_task_ds
from utils.training_utils import load_model_data, load_data, LinePlot, load_args, plot_no_outliers
from pruners.EdgePruner import EdgePruner
from dataset import HFEAPDataset
from huggingface_hub import hf_hub_download

# %%
args = load_args("pruning", 0.001, {
    "desc": "cf",
    "name": "test-ib",
    "window": False,
    "minwindow": 0.5,
    "maxwindow": 2,
    "dataset": "ioi",
    "model": "interp-bench",
    "lr": 1e-3
})

dev_str = 'cuda' if torch.cuda.is_available() else 'mps'

# ablation_types: [mean, resample, cf, oa]
folder, reg_lamb, dataset, ablation_type, dynamic_window = args["folder"], args["lamb"], args["dataset"], args["desc"], args["window"]
model_str = args["model"]
# lr = float(args["lr"])
lr = None
print("Folder", folder)
print("Lamb", reg_lamb)
print("Dataset", dataset)
print("Ablation type", ablation_type)
print("Window", dynamic_window)
print("Learning rate", lr)

if model_str == "gpt2-small":
    model_name = "gpt2-small"
elif model_str == "qwen":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
elif model_str == "interp-bench":
    model_name = "interp-bench"
else:
    raise Exception('Model name not defined')

print(f'Model: {model_name}')

# %%
owt_batch_size = 10

def load_interpbench_model():
    hf_cfg = hf_hub_download("cybershiptrooper/InterpBench", subfolder="ioi", filename="ll_model_cfg.pkl")
    it_model_path = "interpbench/ioi_all_splits/ll_model_100_100_80.pth"

    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        # Some cases in InterpBench have the config as a HookedTransformerConfig object instead of a dict
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = dev_str

    # Small hack to enable evaluation mode in the IOI model, that has a different config during training
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(it_model_path, map_location=dev_str))
    return model

# load model
if model_str == "interp-bench":
    model = load_interpbench_model()
    device, owt_iter = load_data(owt_batch_size, device=dev_str)
    tokenizer = model.tokenizer
else:
    device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size, device=dev_str)

model.train()
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

if model_str == "qwen":
    model.set_ungroup_grouped_query_attention(True)

# node_reg has same units as reg_lamb
# at peak, node_reg adds 50% more regularization to each edge
node_reg = min(0.5 * reg_lamb, 2e-4)

gpu_requeue = True

batch_size = 5

if dataset == "ioi":
    dataset_url = "mech-interp-bench/ioi"
    num_examples = 9500
elif dataset == "arithmetic":
    dataset_url = "mech-interp-bench/arithmetic_addition"
    num_examples = 15000
elif dataset == "arc":
    dataset_url = "mech-interp-bench/arc_easy"
    num_examples = 2100
elif dataset == "greater-than":
    dataset_url = "mech-interp-bench/greater_than"
    num_examples = 13000
elif dataset == "mcqa":
    dataset_url = "mech-interp-bench/copycolors_mcqa"
    num_examples = 100
    batch_size = 2
else:
    raise ValueError("Invalid dataset")


pruning_cfg = EdgeInferenceConfig(model.cfg, device, folder, init_param=1, batch_size=batch_size)
pruning_cfg.lamb = reg_lamb

if reg_lamb <= 1e-4:
    pruning_cfg.lr = 1.5e-1
elif reg_lamb <= 5e-4:
    pruning_cfg.lr = 1e-1
else:
    pruning_cfg.lr = 5e-2

if ablation_type == "cf":
    pruning_cfg.lr /= 5
elif ablation_type == "resample" or ablation_type == "resample_agnostic":
    pruning_cfg.lr /= 2

with open('./hf_token.txt', 'r') as f:
    hf_token = f.read().strip()

hf_dataset = HFEAPDataset(dataset_url, model.tokenizer, task=dataset, split='train', num_examples=num_examples, hf_token=hf_token, model_name=model_name)
if dataset == "mcqa":
    # duplicate the dataset to make it larger
    hf_dataset.dataset = hf_dataset.dataset * 6
dataloader = hf_dataset.to_dataloader(batch_size=batch_size)

for param in model.parameters():
    param.requires_grad = False

# %%
mask_sampler = EdgeMaskUnifSampler(pruning_cfg, node_reg=node_reg)

if dynamic_window:
    mask_sampler.sampling_function = partial(mask_sampler.sample_modified_unif, dynamic_window=True)
    mask_sampler.min_window = args["minwindow"]
    mask_sampler.max_window = args["maxwindow"]
    print(mask_sampler.min_window)
    print(mask_sampler.max_window)

pruner_args = {}
pruner_args['counterfactual_mode'] = True
pruner_args['condition_pos'] = True
edge_pruner = EdgePruner(model, pruning_cfg, mask_sampler, **pruner_args)
edge_pruner.add_cache_hooks()
edge_pruner.add_patching_hooks()

beta2 = 0.995
print(f'LR: {pruning_cfg.lr}')
sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=pruning_cfg.lr, weight_decay=0, betas=(0.9, beta2))

if ablation_type == "oa":
    modal_optimizer = torch.optim.AdamW([edge_pruner.modal_attention, edge_pruner.modal_mlp], lr=pruning_cfg.lr_modes, weight_decay=0)
else:
    modal_optimizer = None
    if not edge_pruner.counterfactual_mode:
        edge_pruner.modal_attention.requires_grad = False
        edge_pruner.modal_mlp.requires_grad = False
# %%

lp_count = pruning_cfg.load_snapshot(edge_pruner, sampling_optimizer, modal_optimizer, gpu_requeue)

take_snapshot = partial(pruning_cfg.take_snapshot, edge_pruner, lp_count, sampling_optimizer, modal_optimizer)
# %%
pruning_cfg.record_every = 100

   
for no_batches, (clean, corrupted, label) in enumerate(tqdm(dataloader)):

    if dataset == "greater-than" and model_str == "gpt2-small":
        label = [[str(y[0] + 1), str(y[0] - 1)] for y in label]
        label = model.tokenizer(label)['input_ids']

        # remove the final 2 digits from the year
        clean = [x[:-2] for x in clean]
        corrupted = [x[:-2] for x in corrupted]

    elif dataset == "greater-than" and model_str == "qwen":
        label = [[str((y[1] + 1) % 10), str((y[1] - 1) % 10)] for y in label]
        label = model.tokenizer(label)['input_ids']

        # remove the final digit from the year
        clean = [x[:-1] for x in clean]
        corrupted = [x[:-1] for x in corrupted]

    plotting = no_batches % (-1 * pruning_cfg.record_every) == -1
    checkpointing = no_batches % (-1 * pruning_cfg.checkpoint_every * pruning_cfg.record_every) == -1

    tokenized_clean = model.tokenizer(clean, return_tensors="pt", padding=True, truncation=True)['input_ids']
    tokenized_corrupted = model.tokenizer(corrupted, return_tensors="pt", padding=True, truncation=True)['input_ids']

    # prepend bos token
    tokenized_clean = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(tokenized_clean.shape[0],1),tokenized_clean], dim=1)
    tokenized_corrupted = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(tokenized_corrupted.shape[0],1),tokenized_corrupted], dim=1)

    # add bos token at the end of the prompt
    tokenized_clean = torch.cat([tokenized_clean, torch.tensor([tokenizer.bos_token_id]).repeat(tokenized_clean.shape[0],1)], dim=1)
    tokenized_corrupted = torch.cat([tokenized_corrupted, torch.tensor([tokenizer.bos_token_id]).repeat(tokenized_corrupted.shape[0],1)], dim=1)

    # last_token_pos is the last token position in the prompt (NOT the label position). For IOI, I believe names are guaranteed to be a single token long
    last_token_pos = (((tokenized_clean != tokenizer.pad_token_id) & (tokenized_clean != tokenizer.eos_token_id)) * torch.arange(tokenized_clean.shape[1])).argmax(dim=-1) - 1
    last_token_pos_corrupted = (((tokenized_corrupted != tokenizer.pad_token_id) & (tokenized_corrupted != tokenizer.eos_token_id)) * torch.arange(tokenized_corrupted.shape[1])).argmax(dim=-1) - 1

    # add label at last token position for clean and corrupted
    labels_clean = torch.tensor([label]).squeeze()[:,0]
    labels_corrupted = torch.tensor([label]).squeeze()[:,1]

    # add 1 to last token pos to get the label position
    last_token_pos += 1
    last_token_pos_corrupted += 1

    tokenized_clean[torch.arange(tokenized_clean.shape[0]), last_token_pos + 1] = labels_clean
    tokenized_corrupted[torch.arange(tokenized_corrupted.shape[0]), last_token_pos_corrupted + 1] = labels_corrupted

    batch = tokenized_clean
    cf = tokenized_corrupted

    assert (last_token_pos == last_token_pos_corrupted).all(), "Last token positions do not match"

    sampling_optimizer.zero_grad()
    if ablation_type == "oa":
        modal_optimizer.zero_grad()

    # sample prune mask
    graph_suffix = f"-{no_batches}" if checkpointing else "" if plotting else None
    
    loss = edge_pruner(batch.to(device), last_token_pos.to(device), cf.to(device), graph_suffix=graph_suffix, timing=False)

    loss.backward()

    grad_norms = mask_sampler.clip_grad(5)

    prev_alphas = mask_sampler.get_sampling_params()[:,0].detach().clone()

    sampling_optimizer.step()

    if ablation_type == "oa":
        prev_modes = edge_pruner.get_modes().detach().clone()
        modal_optimizer.step()

    if dynamic_window:
        # update param vars for dynamic window
        optim_state = sampling_optimizer.state_dict()['state']
        mask_sampler.update_param_vars([optim_state[x]['exp_avg_sq'] / (1 - beta2 ** optim_state[x]['step']) for x in optim_state])
    
    mask_sampler.fix_nans()

    with torch.no_grad():
        step_sz = (mask_sampler.get_sampling_params()[:,0] - prev_alphas).abs()
        step_sz = (step_sz - 1e-3).relu().sum() / (step_sz > 1e-3).sum()
        lp_entry = {
            "step_size": step_sz.item(), 
            "max_grad_norm": np.max(grad_norms)
        }

        if ablation_type == "oa":
            mode_step_sz = (edge_pruner.get_modes().clone() - prev_modes).norm(dim=-1).mean()
            lp_entry["mode_step_size"] = mode_step_sz.item()
            
        lp_count.add_entry(lp_entry)

    if plotting:
        take_snapshot("")
        if checkpointing:
            take_snapshot(f"-{no_batches}")
# %%
