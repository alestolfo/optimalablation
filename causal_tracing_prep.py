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
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.training_utils import load_model_data, LinePlot
from torch.utils.data import DataLoader
from utils.tracing_utils import get_subject_tokens

# %%

ds_path = "utils/datasets/facts"
# get commands to download data
if not os.path.exists(f"{ds_path}/attributes_1.txt"):
    dirs = {}
    with open(f"{ds_path}/attributes.txt", "r") as f:
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
        st += f"mkdir {ds_path}/{subdir};\n "
        for file in dirs[c]:
            st += f"wget -P {ds_path}/{subdir} {c}/{file};\n "

    with open(f"{ds_path}/attributes_1.txt", "w") as f:
        f.write(st)

# %%

# combine data files
if not os.path.exists(f"{ds_path}/attributes_ds.pkl"):
    fact_ds = []
    for path in glob(f"{ds_path}/*/*.json"):
        _, ds_type, category = path.rsplit("/", 2) 
        with open(path, "r") as f:
            ds = json.load(f)
        for template in [*ds['prompt_templates'], *ds['prompt_templates_zs']]:
            for sample in ds['samples']:
                fact_ds.append({'relation_type': ds['properties']['relation_type'], 'relation_name': ds['name'], 'template': template, 'subject': sample['subject'], 'object': sample['object']})

                if ds['properties']['symmetric']:
                    fact_ds.append({'relation_type': ds['properties']['relation_type'], 'relation_name': ds['name'], 'template': template, 'subject': sample['object'], 'object': sample['subject']})

    with open(f"{ds_path}/attributes_ds.pkl", 'wb') as f:
        pickle.dump(fact_ds, f)
else:
    with open(f"{ds_path}/attributes_ds.pkl", 'rb') as f:
        attribute_ds = pickle.load(f)
# %%
with open(f"{ds_path}/known_1000.json", 'rb') as f:
    known_facts = json.load(f)

# %%

# filter for correct prompts

sns.set()
mode = "fact"
folder="results/causal_tracing"
# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-xl"
batch_size = 40
clip_value = 1e5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)
tokenizer = model.tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model

# %%
if mode == "attribute":
    data_loader = DataLoader(attribute_ds, batch_size=batch_size)
    data_iter = iter(data_loader)
else:
    data_loader = DataLoader(known_facts, batch_size=batch_size)
    data_iter = iter(data_loader)

# %%
answer_candidates = 3
# rejection_prob = 0.1

kl_loss = torch.nn.KLDivLoss(reduction="none")
# %%

out_file = "my_attributes" if mode == "attribute" else "my_known"
ans_key = "object" if mode == "attribute" else "attribute"

correct = 0
wrong = 0
my_dataset = []

for batch in tqdm(data_iter):
    
    tokens, _ = get_subject_tokens(batch, tokenizer, mode=mode)
    predictions = model.generate(tokens, temperature=0, max_new_tokens=5)
    # decode the new tokens
    predictions = tokenizer.batch_decode(predictions[:, tokens.shape[-1]:])
    for i, (pred, answer) in enumerate(zip(predictions, batch[ans_key])):
        pattern = r'^[^a-zA-Z]+'
        pred = re.sub(pattern, '', pred)
        answer = re.sub(pattern, '', answer)

        if answer.startswith(pred) or pred.startswith(answer):
            correct += 1
            my_dataset.append({k: batch[k][i] for k in batch})
        else:
            wrong += 1

    print((correct / (correct + wrong)))
            # print("Prompt:", batch['prompt'][i])
            # print("Pred:", pred)
            # print("Ans:", batch['prediction'][i])
            # print("Ans:", answer)

print(correct)
print(wrong)
# %%
with open(f"{ds_path}/{out_file}.pkl", "wb") as f:
    pickle.dump(my_dataset, f)

# %%

# combine facts and attributes ds
# with open("utils/datasets/facts/my_attributes.pkl", "rb") as f:
#     attributes_ds = pickle.load(f)

# correct_prompts = []
# subject_object_pairs = set()
# rel_names = {}
# for a_line in attributes_ds:
#     a_line["attribute"] = a_line["object"]
#     del a_line["object"]
#     if (a_line["subject"], a_line["attribute"]) in subject_object_pairs:
#         # print("dupe")
#         continue
#     if a_line['relation_name'] not in rel_names:
#         rel_names[a_line['relation_name']] = 1
#     else:
#         rel_names[a_line['relation_name']] += 1
#     # if rel_names[a_line['relation_name']] >= 200:
#     #     continue
#     subject_object_pairs.add((a_line["subject"], a_line["attribute"]))
#     correct_prompts.append({"info": a_line["relation_name"], "subject": a_line["subject"], "object": a_line["attribute"], "prompt": a_line["prompt"].replace("{}", a_line["subject"])})

# # %%
# with open(f"{ds_path}/my_known.pkl", "rb") as f:
#     facts_ds = pickle.load(f)

# # %%
# for a_line in facts_ds:
#     if (a_line["subject"], a_line["attribute"]) in subject_object_pairs:
#         print("dupe")
#         print(a_line["subject"], a_line["attribute"])
#         continue
#     subject_object_pairs.add((a_line["subject"], a_line["attribute"]))
#     correct_prompts.append({"info": a_line["relation_id"], "subject": a_line["subject"], "object": a_line["attribute"], "prompt": a_line["prompt"]})


# # %%
# with open("utils/datasets/facts/my_facts.pkl", "wb") as f:
#     pickle.dump(correct_prompts, f)
