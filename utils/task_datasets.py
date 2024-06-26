# %%
import torch
import datasets
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import os
import random
import time
from itertools import cycle
import pickle
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils.training_utils import load_model_data, LinePlot, gen_resample_perm
from utils.datasets.ioi.ioi_dataset import IOIDataset
from utils.datasets.greater_than.utils import get_valid_years
from utils.datasets.greater_than.data import YearDataset

class TaskDataset(Dataset):
    def __init__(self, data, last_token_pos, cf=None):
        assert data.shape[0] == last_token_pos.shape[0]

        self.data = data
        self.last_token_pos = last_token_pos

        if cf is not None:
            assert cf.shape[0] == data.shape[0]
        
        self.cf = cf

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.cf is None:
            return self.data[index], self.last_token_pos[index]
        else:
            return self.data[index], self.last_token_pos[index], self.cf[index]

class TaskConfig():
    # generally: zero, mean, mean_agnostic, resample, resample_agnostic, cf_mean, cf, oa
    def __init__(self, ds_name, batch_size, device, ablation_type="oa"):
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.device = device

        self.ablation_type = ablation_type
        self.means = None
        self.ds = iter([])
    
    def get_pruner_args(self, ablation_types={"mean", "resample", "cf", "oa"}):
        if self.ablation_type not in ablation_types:
            raise Exception(f"ablation type {self.ablation_type} not allowed")
        
        pruner_args = {}
        if self.ablation_type == "zero":
            zero_modes = []
            init_modes = self.init_modes()
            for ts in init_modes:
                zero_modes.append(torch.zeros_like(ts).to(ts.device))
            pruner_args['init_modes'] = zero_modes
        elif self.ablation_type == "mean" or self.ablation_type == "cf_mean":
            pruner_args['init_modes'] = self.init_modes(None)
            pruner_args['condition_pos'] = True
        elif self.ablation_type == "mean_agnostic":
            pruner_args['init_modes'] = self.init_modes(1)
        elif self.ablation_type == "resample" or self.ablation_type == "cf":
            pruner_args['counterfactual_mode'] = True
            pruner_args['condition_pos'] = True
        elif self.ablation_type == "resample_agnostic":
            pruner_args['counterfactual_mode'] = True
        elif self.ablation_type == "oa":
            pruner_args['init_modes'] = self.init_modes()
        elif self.ablation_type == "oa_specific":
            pruner_args['init_modes'] = self.init_modes(None)
            pruner_args['condition_pos'] = True
        else:
            raise Exception(f"ablation type {self.ablation_type} not supported")
        
        return pruner_args

    def process_means(self, all_means, samples, cutoff=None):
        if cutoff:
            min_length = cutoff
        else:
            min_length = (torch.arange(samples.shape[0]).to(self.device) * (samples == samples.max())).argmax().item()

        processed_means = []
        for means in all_means:
            # [seq_pos, layer, ...]
            s = samples[(..., *[None for _ in means.shape[1:]])]
            
            general_mean = (means[min_length:] * s[min_length:]).sum(dim=0) / s[min_length:].sum()
            processed_means.append(
                torch.cat((means[:min_length],general_mean[None, :]), dim=0)
                if cutoff is None else general_mean
            )
        return processed_means

    def init_modes(self, cutoff=9):
        cf_tag = "cf_" if self.ablation_type == "cf_mean" else ""

        with open(f"results/oca/{self.ds_name}/means_{cf_tag}attention.pkl", "rb") as f:
            #  seq_len x n_layers x n_heads x d_head
            init_modes_attention = pickle.load(f)
        with open(f"results/oca/{self.ds_name}/means_{cf_tag}mlp.pkl", "rb") as f:
            # seq_len x n_layers x d_model
            init_modes_mlp = pickle.load(f)
        with open(f"results/oca/{self.ds_name}/means_{cf_tag}samples.pkl", "rb") as f:
            # seq_len
            samples = pickle.load(f)

        return self.process_means([init_modes_attention, init_modes_mlp], samples, cutoff=cutoff)
    
    def gen_ds(self, tokenizer):
        pass

    def retrieve_batch_cf(self, tokenizer):
        batch_data = next(self.ds, None)
        if batch_data is None:
            self.gen_ds(tokenizer)
            batch_data = next(self.ds)

        batch = batch_data[0]
        last_token_pos = batch_data[1]
        cf = None

        if self.ablation_type.startswith("resample"):
            permutation = gen_resample_perm(batch.shape[0])

            cf = batch[permutation]
            # if resampled sequence i shorter than original sequence, move padding to left
            padding_left = last_token_pos - last_token_pos[permutation]
            for i in range(batch.shape[0]):
                if padding_left[i] > 0:
                    cf[i] = torch.cat((cf[i,-padding_left[i]:], cf[i, :-padding_left[i]]), dim=-1)
            cf = cf.to(self.device)
            
        elif self.ablation_type == "cf":
            cf = batch_data[2].to(self.device)

        return batch.to(self.device), last_token_pos.int().to(self.device), cf
        
# class OWTConfig():
#     def __init__(self, owt_iter, device):
#         self.ds_iter = owt_iter
#         self.device = device
    
#     def next_batch(self, tokenizer=None):
#         # BOS is already prepended
#         batch = next(self.ds_iter)['tokens'].to(self.device)
#         return batch, batch.shape[1] - 1

class IOIConfig(TaskConfig):
    def __init__(self, batch_size, device, ablation_type="oa", fix_prompt=False, test=False):
        super().__init__("ioi", batch_size, device, ablation_type)

        self.seed = 293088429 if test else 0
        self.fix_prompt = fix_prompt

    def gen_ds(self, tokenizer):
        ioi_dataset = IOIDataset(
            prompt_type="ABBA",
            N=self.batch_size * 100,
            # if fix prompt, output only one prompt template per batch to enable resamples
            nb_templates=random.randint(1,15) if self.fix_prompt else None,
            single_template=self.fix_prompt,
            seed=self.seed
        )
        self.seed += 1
        data = ioi_dataset.toks
        # prepend bos token
        data = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(data.shape[0],1),data], dim=1)

        # last_token_pos is the last token position in the prompt (NOT the label position). For IOI, I believe names are guaranteed to be a single token long
        last_token_pos = ((data != tokenizer.pad_token_id) * torch.arange(data.shape[1])).argmax(dim=-1) - 1

        if self.ablation_type == "cf":
            cf = (
                ioi_dataset
                .gen_flipped_prompts(("IO", "RAND"), seed=1)
                .gen_flipped_prompts(("S", "RAND"), seed=2)
                .gen_flipped_prompts(("S1", "RAND"), seed=3)
            ).toks
            cf = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(cf.shape[0],1),cf], dim=1)
            
            ds = TaskDataset(data, last_token_pos, cf)
        else:
            ds = TaskDataset(data, last_token_pos)
        
        self.ds = iter(DataLoader(ds, self.batch_size, shuffle=True))
        
class GTConfig(TaskConfig):
    def __init__(self, batch_size, device, ablation_type="oa", test=False):
        super().__init__("gt", batch_size, device, ablation_type)

        self.years_to_sample_from = None

    def gen_ds(self, tokenizer):
        if self.years_to_sample_from is None:
            self.years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
        gt_dataset = YearDataset(
            self.years_to_sample_from, 
            self.batch_size * 100, 
            Path("utils/datasets/greater_than/potential_nouns.txt"), 
            tokenizer, balanced=False, device=self.device, eos=False)

        data = gt_dataset.good_toks

        # prepend bos token. Batch does not contain labels
        data = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(data.shape[0],1).to(self.device),data], dim=1)

        # last_token_pos is the last token position in the prompt (NOT the label position)
        last_token_pos = ((data.shape[1] - 1) * torch.ones(data.shape[0])).int().to(self.device)
        
        # examples with start year replaced with "01"
        if self.ablation_type == "cf":
            cf = gt_dataset.bad_toks
            cf = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(cf.shape[0],1).to(self.device),cf], dim=1)

            ds = TaskDataset(data, last_token_pos, cf)
        else:
            ds = TaskDataset(data, last_token_pos)
        
        self.ds = iter(DataLoader(ds, self.batch_size, shuffle=True))

# class ColorConfig():
#     def __init__(self, batch_size, device):
#         self.batch_size = batch_size
#         self.device = device
        
#         with open("color_objects/task.json") as f:
#             color_ds = json.load(f)

#         self.ds_iter = cycle(color_ds['examples'][:1500])
        
#     def next_batch(self, tokenizer):
#         batch = tokenizer(["Q: " + next(self.ds_iter)['input'] + " A: It's a" for _ in range(self.batch_size)], padding=True, return_tensors='pt')['input_ids'].to(self.device)
#         last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(self.device)).argmax(dim=-1)
#         return batch, last_token_pos

def get_task_ds(dataset, bsz, device, ablation_type="oa", fix_prompt=False):
    if dataset == "ioi":
        task_ds = IOIConfig(bsz, device, ablation_type, fix_prompt=fix_prompt)
    elif dataset == "gt":
        task_ds = GTConfig(bsz, device, ablation_type)
    else:
        raise Exception(f"Dataset {dataset} not defined")
    return task_ds
