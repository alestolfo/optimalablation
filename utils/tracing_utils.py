import torch

# subject token pos is a [X by 2] tensor containing ALL positions of subject tokens (could be multiple subject tokens per sample)
def replace_subject_tokens(bsz, subject_token_pos, null_token, act, hook):
    act = act.unflatten(0, (-1, bsz))

    # first is clean
    for j in range(null_token.shape[0]):
        act[j+1, subject_token_pos[:,0], subject_token_pos[:,1]] = null_token[j].clone()
    
    return act.flatten(start_dim=0, end_dim=1)

# subject token pos is a [X by 2] tensor containing ALL positions of subject tokens (could be multiple subject tokens per sample)
def gauss_subject_tokens(bsz, subject_token_pos, std, act, hook):
    act = act.unflatten(0, (-1, bsz))

    # first is clean
    for j in range(1, act.shape[0]):
        act[j, subject_token_pos[:,0], subject_token_pos[:,1]] += torch.randn(size=(subject_token_pos.shape[0], act.shape[-1])).to(act.device) * std

    return act.flatten(start_dim=0, end_dim=1)

# def copy_corrupted_hook(bsz, act, hook):
#     act = torch.cat([act, act[bsz:(2 * bsz)]], dim=0)
#     print(act.shape)
#     return act

def patch_component_last_token(bsz, layer_idx, act, hook):
    act = act.unflatten(0, (-1, bsz))
    act[layer_idx+1, :, -1] = act[0, :, -1].clone()
    return act.flatten(start_dim=0, end_dim=1)

def patch_component_subject_tokens(bsz, layer_idx, subject_token_pos, act, hook):
    act = act.unflatten(0, (-1, bsz))
    act[layer_idx+1, subject_token_pos[:,0], subject_token_pos[:,1]] = act[0, subject_token_pos[:,0], subject_token_pos[:,1]].clone()
    return act.flatten(start_dim=0, end_dim=1)

def patch_component_all_tokens(bsz, layer_idx, act, hook):
    act = act.unflatten(0, (-1, bsz))
    act[layer_idx+1] = act[0].clone()
    return act.flatten(start_dim=0, end_dim=1)

# batch has a "prompt" and "subject" column
def get_subject_tokens(batch, tokenizer, mode="fact"):
    if mode == "attribute":
        batch['prompt'] = [template.replace("{}", subject) for template, subject in zip(batch['template'], batch['subject'])]

    subject_pos = []
    for i, (prompt, subject) in enumerate(zip(batch['prompt'], batch['subject'])): 
        pre_subject = prompt.split(subject)
        
        if len(pre_subject) == 1:
            print("NOT EXPECTED: SUBJECT NOT FOUND")
        # assert len(pre_subject) > 1
        pre_subject = pre_subject[0]
        subject_pos.append([len(pre_subject), len(pre_subject) + len(subject)])

    # bsz x 2
    subject_pos = torch.tensor(subject_pos).unsqueeze(1)
    tokens = tokenizer(batch['prompt'], padding=True, return_tensors='pt', return_offsets_mapping=True)

    # tokens['offset_mapping']: batch x seq_pos x 2
    subject_tokens = ((
        # start or end char falls between beginning and end of subject
        (tokens['offset_mapping'] > subject_pos[...,[0]]) * 
        (tokens['offset_mapping'] < subject_pos[...,[1]])
    ) + (
        # end char equals end char of subject, or start char equals start char of subject
        (tokens['offset_mapping'] == subject_pos) * 
        # except for EOT, 
        (tokens['offset_mapping'][...,[1]] - tokens['offset_mapping'][...,[0]])
    )).sum(dim=-1).nonzero()

    return tokens['input_ids'], subject_tokens
