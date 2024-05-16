# %%
import torch
import json
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from glob import glob
from functools import partial
import os
import torch.optim
import time
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, LinePlot
from torch.utils.data import DataLoader

# %%
# get commands to download data
if not os.path.exists("utils/datasets/facts/attributes_1.txt"):
    dirs = {}
    with open("utils/datasets/facts/attributes.txt", "r") as f:
        factual_dirs = f.readlines()

    new_category = False
    cur_category = ""
    for s in factual_dirs:
        if s == "\n":
            new_category = True
            continue
        if new_category:
            cur_category = s[:-1]
            dirs[cur_category] = []
            new_category = False
            continue
        dirs[cur_category].append(s.split("\t",1)[0].split("]")[-1])

    st = ""
    for c in dirs:
        subdir = c[:-1].rsplit("/",1)[-1]
        st += f"mkdir utils/datasets/facts/{subdir};\n "
        for file in dirs[c]:
            st += f"wget -P utils/datasets/facts/{subdir} {c}/{file};\n "

    with open("utils/datasets/facts/attributes_1.txt", "w") as f:
        f.write(st)

# %%

# combine data files
if not os.path.exists('utils/datasets/facts/attributes_ds.pkl'):
    fact_ds = []
    for path in glob("utils/datasets/facts/*/*.json"):
        _, ds_type, category = path.rsplit("/", 2) 
        with open(path, "r") as f:
            ds = json.load(f)
            for template in [*ds['prompt_templates'], *ds['prompt_templates_zs']]:
                for sample in ds['samples']:
                    fact_ds.append({'relation_type': ds['properties']['relation_type'], 'relation_name': ds['name'], 'template': template, 'subject': sample['subject'], 'object': sample['object']})

                    if ds['properties']['symmetric']:
                        fact_ds.append({'relation_type': ds['properties']['relation_type'], 'relation_name': ds['name'], 'template': template, 'subject': sample['object'], 'object': sample['subject']})

    with open('utils/datasets/facts/attributes_ds.pkl', 'wb') as f:
        pickle.dump(fact_ds, f)
else:
    with open('utils/datasets/facts/attributes_ds.pkl', 'rb') as f:
        fact_ds = pickle.load(f)
# %%

# filter for correct prompts

sns.set()

folder="results/causal_tracing"
# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-xl"
batch_size = 20
clip_value = 1e5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)
tokenizer = model.tokenizer

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

# %%
data_loader = DataLoader(fact_ds, batch_size=batch_size)
data_iter = iter(data_loader)

# %%
answer_candidates = 3
rejection_prob = 0.1

# %%
with torch.no_grad():
    correct_prompts = []
    for batch in tqdm(data_iter):
        prompts = []
        for i, temp in enumerate(batch['template']):
            prompts.append(temp.replace("{}", batch['subject'][i]))
        input = tokenizer(prompts, padding=True, return_tensors='pt')['input_ids']
        last_token_pos = ((input != tokenizer.pad_token_id) * torch.arange(input.shape[-1]).repeat(input.shape[0], 1)).argmax(dim=-1).to(device)
        for i in range(len(input)):
            offset = input.shape[-1] - last_token_pos[i] - 1
            if offset > 0:
                input[i] = torch.cat((input[i,-offset:], input[i,:-offset]), dim=-1)
        input = input.to(device)

        output_probs = model(input)[:,-1,:].softmax(dim=-1)

        top_probs, top_idx = output_probs.topk(answer_candidates, dim=-1)
        for top_x in range(answer_candidates):
            print(top_x)
            self_prompt = torch.cat((input, top_idx[:, [top_x]]), dim=-1)
            prediction = model.generate(self_prompt, temperature=0, max_new_tokens=5)
            answers = tokenizer.batch_decode(prediction[:, input.shape[-1]:])

            for i, answer in enumerate(answers):
                print(prompts[i], "|", answer, "|", batch['object'][i])
                if (top_probs[i, top_x] > rejection_prob and (
                    answer.strip().startswith(batch['object'][i]) 
                    or batch['object'][i].startswith(answer.strip())
                )):
                    correct_prompts.append({"rel": batch['relation_name'], "template": batch['template'][i], "subject": batch['subject'][i], "object": batch['object'][i]})
    
with open(f"utils/datasets/facts/correct_facts_{model_name}.pkl", 'wb') as f:
    pickle.dump(correct_prompts, f)
