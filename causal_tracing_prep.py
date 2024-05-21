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
mode = "attribute"
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

out_file = "my_attributes" if mode == "attribute" else "my_facts"
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


# %%


# correct_count = 0
# with torch.no_grad():
#     correct_prompts = []
#     for batch in tqdm(data_iter):

#         prompts = batch['prompt']
            
#         prompts = []
#         for i, temp in enumerate(batch['template']):
#             prompts.append(temp.replace("{}", batch['subject'][i]))
        
#         input = tokenizer(prompts, padding=True, return_tensors='pt')
#         input = input['input_ids'].to(device)

#         prediction = model.generate(input, temperature=0, max_new_tokens=5)
#         answers = tokenizer.batch_decode(prediction[:, input.shape[-1]:])

#         for i, answer in enumerate(answers):
#             print(prompts[i], "|", answer, "|", batch['object'][i])
#             correct = answer.strip().startswith(batch['object'][i]) or batch['object'][i].startswith(answer.strip())
#             if correct:
#                 correct_count += 1

            # if (top_probs[i, top_x] > rejection_prob and (
            #     answer.strip().startswith(batch['object'][i]) 
            #     or batch['object'][i].startswith(answer.strip())
            # )):
            #     correct_prompts.append({"rel": batch['relation_name'][i], "template": batch['template'][i], "subject": batch['subject'][i], "object": batch['object'][i]})


        # output_probs = model(input['input_ids'])[:,-1,:].softmax(dim=-1)

        # top_probs, top_idx = output_probs.topk(answer_candidates, dim=-1)
        # for top_x in range(answer_candidates):
        #     print(top_x)
        #     self_prompt = torch.cat((input, top_idx[:, [top_x]]), dim=-1)
        #     prediction = model.generate(self_prompt, temperature=0, max_new_tokens=5)
        #     answers = tokenizer.batch_decode(prediction[:, input.shape[-1]:])

        #     for i, answer in enumerate(answers):
        #         print(prompts[i], "|", answer, "|", batch['object'][i])
        #         if (top_probs[i, top_x] > rejection_prob and (
        #             answer.strip().startswith(batch['object'][i]) 
        #             or batch['object'][i].startswith(answer.strip())
        #         )):
        #             correct_prompts.append({"rel": batch['relation_name'][i], "template": batch['template'][i], "subject": batch['subject'][i], "object": batch['object'][i]})

# %%

# another way to do this: tokenize the answer and then pick those with high enough prediction prob

# another way to do this: check which tokens are compatible with the answer
# 

# with torch.no_grad():
#     bos_output = model(input['input_ids'], padding_side="left")[:,-1].softmax(dim=-1)
#     n_bos_output = model(input['input_ids'])[:,-1].softmax(dim=-1)
# # %%
# out_file = f"utils/datasets/facts/correct_facts_{model_name}.pkl"
# if not os.path.exists(out_file):
#     with open(out_file, 'wb') as f:
#         pickle.dump(correct_prompts, f)

# %%
