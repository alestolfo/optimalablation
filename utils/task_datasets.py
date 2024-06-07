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
import torch.optim
import os
import random
import time
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from utils.training_utils import load_model_data, LinePlot, gen_resample_perm
from utils.datasets.ioi.ioi_dataset import IOIDataset
from utils.datasets.greater_than.utils import get_valid_years
from utils.datasets.greater_than.data import YearDataset

class TaskDataset():
    def __init__(self, ds_name, batch_size, device, counterfactual=False):
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.device = device
        self.counterfactual = counterfactual

    def process_means(self, all_means, samples, cutoff=None):
        if cutoff:
            min_length = cutoff
        else:
            min_length = (torch.arange(samples.shape[0]).to(self.device) * (samples == samples.max())).argmax().item()

        processed_means = []
        for means in all_means:
            general_mean = (means[min_length:].permute(0,-1) * samples[min_length:]).sum(dim=-1) / samples[min_length].sum()
            processed_means.append(
                (torch.cat((means[:min_length],general_mean[:, None]), dim=0)
                 if cutoff is None else general_mean).transpose(0,1)
            )
        return processed_means

    def init_modes(self, cf=False, oa_init=False):
        if cf:
            cf_tag = "cf_"
        else:
            cf_tag = ""

        with open(f"results/oca/{self.ds_name}/means_{cf_tag}attention.pkl", "rb") as f:
            #  seq_len x n_layers x n_heads x d_head
            init_modes_attention = pickle.load(f)
        with open(f"results/oca/{self.ds_name}/means_{cf_tag}mlp.pkl", "rb") as f:
            # seq_len x n_layers x d_model
            init_modes_mlp = pickle.load(f)
        with open(f"results/oca/{self.ds_name}/means_{cf_tag}samples.pkl", "rb") as f:
            # seq_len
            samples = pickle.load(f)
            
        return self.process_means([init_modes_attention, init_modes_mlp], samples, cutoff=9 if oa_init else None)

    def next_batch(self, tokenizer, test=False, counterfactual=False):
        return None, None, None

    def retrieve_batch_cf(self, tokenizer, ablation_type, test=False):
        batch_data = self.next_batch(tokenizer, test=test)
        batch = batch_data[0]
        last_token_pos = batch_data[1]
        cf = batch_data[2] if ablation_type == "cf" else None

        if ablation_type == "resample":
            permutation = gen_resample_perm(batch.shape[0]).to(self.device)

            cf = batch[permutation]
            # if resampled sequence i shorter than original sequence, move padding to left
            padding_left = last_token_pos - last_token_pos[permutation]
            for i in range(batch.shape[0]):
                if padding_left[i] > 0:
                    cf[i] = torch.cat((cf[i,-padding_left[i]:], cf[i, :-padding_left[i]]), dim=-1)
        
        return batch, last_token_pos.int(), cf
        
class OWTConfig():
    def __init__(self, owt_iter, device):
        self.ds_iter = owt_iter
        self.device = device
    
    def next_batch(self, tokenizer=None):
        # BOS is already prepended
        batch = next(self.ds_iter)['tokens'].to(self.device)
        return batch, batch.shape[1] - 1
    
# class IOIConfigDiverse(TaskDataset):
#     def __init__(self, batch_size, device, test_size=10000):
#         super().__init__(batch_size, device)
        
#         ioi_ds = datasets.load_from_disk("../plausibleablation/data/ioi/ioi")

#         generator = torch.Generator().manual_seed(6942)
#         test_set, train_set = torch.utils.data.random_split(ioi_ds['train'], [test_size,len(ioi_ds['train']) - test_size], generator=generator)

#         ioi_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
#         ioi_iter = cycle(iter(ioi_loader))

#         self.ds_iter = ioi_iter
#         self.ds_test = DataLoader(test_set, batch_size=batch_size)
#         self.test_iter = cycle(iter(self.ds_test))
        
#     def init_modes(self):
#         with open("results/oca/ioi/means_attention.pkl", "rb") as f:
#             #  n_layers x 10 (seq_len) x n_heads x d_head
#             init_modes_attention = pickle.load(f)
#         with open("results/oca/ioi/means_mlp.pkl", "rb") as f:
#             #  n_layers x 10 (seq_len) x d_model
#             init_modes_mlp = pickle.load(f)
#         return init_modes_attention[:,-1], init_modes_mlp[:,-1]

#     def next_batch(self, tokenizer, test=False, counterfactual=False):
#         b = next(self.test_iter if test else self.ds_iter)

#         # remove label, it can be more than one token
#         # batch = [s.rsplit(' ', 1)[0] for s in b['ioi_sentences']]
#         batch = b['ioi_sentences']

#         batch = tokenizer(batch, padding=True, return_tensors='pt')['input_ids'].to(self.device)
        
#         # prepend bos token
#         batch = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(batch.shape[0],1).to(self.device),batch], dim=1)

#         # last_token_pos is the last token position in the prompt (NOT the label position)
#         last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(self.device)).argmax(dim=-1) - 1
        
#         return batch, last_token_pos

class IOIConfig(TaskDataset):
    def __init__(self, batch_size, device, counterfactual=False, fix_prompt=False):
        super().__init__("ioi", batch_size, device, counterfactual)

        self.seed = 0
        self.fix_prompt = fix_prompt
        self.ds = None

    def gen_ds(self):
        cf = (
            ioi_dataset
            .gen_flipped_prompts(("IO", "RAND"), seed=1)
            .gen_flipped_prompts(("S", "RAND"), seed=2)
            .gen_flipped_prompts(("S1", "RAND"), seed=3)
        ).toks
        cf = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(cf.shape[0],1),cf], dim=1).to(self.device)


    def next_batch(self, tokenizer, test=False):
        ioi_dataset = IOIDataset(
            prompt_type="ABBA",
            N=self.batch_size,
            # if fix prompt, output only one prompt template per batch to enable resamples
            nb_templates=random.randint(1,15) if self.fix_prompt else None,
            single_template=self.fix_prompt,
            seed=self.seed + (293088429 if test else 0)
        )
        self.seed += 1
        batch = ioi_dataset.toks

        # prepend bos token
        batch = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(batch.shape[0],1),batch], dim=1).to(self.device)

        # last_token_pos is the last token position in the prompt (NOT the label position). In this dataset, I believe names are guaranteed to be a single token long
        last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(self.device)).argmax(dim=-1) - 1

        if counterfactual:
            return batch, last_token_pos, cf
        else:
            return batch, last_token_pos

class GTConfig(TaskDataset):
    def __init__(self, batch_size, device, counterfactual=False):
        super().__init__("gt", batch_size, device, counterfactual)

        self.years_to_sample_from = None

    def next_batch(self, tokenizer, test=False, counterfactual=False):
        if self.years_to_sample_from is None:
            self.years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
        batch = YearDataset(
            self.years_to_sample_from, 
            self.batch_size, 
            Path("utils/datasets/greater_than/potential_nouns.txt"), 
            tokenizer, balanced=False, device=self.device, eos=False)

        # examples with start year replaced with "01"
        if counterfactual:
            cf = batch.bad_toks
        batch = batch.good_toks

        # prepend bos token. Batch does not contain labels
        batch = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(batch.shape[0],1).to(self.device),batch], dim=1)

        # last_token_pos is the last token position in the prompt (NOT the label position)
        last_token_pos = ((batch.shape[1] - 1) * torch.ones(batch.shape[0])).int().to(self.device)
        
        if counterfactual:
            cf = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(cf.shape[0],1).to(self.device),cf], dim=1)
            return batch, last_token_pos, cf
        else:
            return batch, last_token_pos

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

def get_task_ds(dataset, bsz, device, fix_prompt=False):
    if dataset == "ioi":
        task_ds = IOIConfig(bsz, device, fix_prompt=fix_prompt)
    # elif dataset == "ioi_b":
    #     task_ds = IOIConfigDiverse(bsz, device)
    elif dataset == "gt":
        task_ds = GTConfig(bsz, device)
    else:
        raise Exception(f"Dataset {dataset} not defined")
    return task_ds
