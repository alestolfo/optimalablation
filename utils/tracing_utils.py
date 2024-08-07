import torch
from functools import partial

attn_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out" 
mlp_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out" 

# subject token pos is a [X by 2] tensor containing ALL positions of subject tokens (could be multiple subject tokens per sample)
def replace_subject_tokens(bsz, subject_token_pos, null_token, act, hook):
    act = act.unflatten(0, (-1, bsz))

    # diff null tokens for each
    if null_token.shape[0] > 1:
        # first is clean
        for j in range(null_token.shape[0]):
            act[j+1, subject_token_pos[:,0], subject_token_pos[:,1]] = null_token[j].clone()
    
    # same null token for everything
    else:
        for j in range(1, act.shape[0]):
            act[j, subject_token_pos[:,0], subject_token_pos[:,1]] = null_token.clone()
    
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

def patch_component_last_token(bsz, layer_idx, window_size, act, hook):
    act = act.unflatten(0, (-1, bsz))
    if window_size == 0:
        act[layer_idx+1, :, -1] = act[0, :, -1].clone()
    else:
        for layer in range(
            max(0, layer_idx-window_size),

            # last is fully corrupted, don't mess with it
            min(act.shape[0]-1, layer_idx+window_size+1)
        ):
            act[layer+1, :, -1] = act[0, :, -1].clone()
    return act.flatten(start_dim=0, end_dim=1)

def patch_component_token_pos(bsz, layer_idx, subject_token_pos, window_size, act, hook):
    act = act.unflatten(0, (-1, bsz))
    if window_size == 0:
        act[layer_idx+1, subject_token_pos[:,0], subject_token_pos[:,1]] = act[
            0, subject_token_pos[:,0], subject_token_pos[:,1]
        ].clone()
    else:
        for layer in range(
            max(0, layer_idx-window_size),

            # last is fully corrupted, don't mess with it
            min(act.shape[0]-1, layer_idx+window_size+1)
        ):
            act[layer+1, subject_token_pos[:,0], subject_token_pos[:,1]] = act[
                0, subject_token_pos[:,0], subject_token_pos[:,1]
            ].clone()
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

def get_patch_token_pos(tokens, subject_pos, device, token_type):
    if token_type == "last":
        patch_token_pos = None
    elif token_type == "last_subject":
        mask = torch.zeros_like(tokens).to(device)
        mask[subject_pos[:,0], subject_pos[:,1]] = 1
        last_subject_pos = (mask * torch.arange(mask.shape[-1]).to(device)).argmax(dim=-1)
        patch_token_pos = torch.stack([torch.arange(mask.shape[0]).to(device), last_subject_pos], dim=-1)
    elif token_type == "all_subject":
        patch_token_pos = subject_pos
    
    return patch_token_pos

def ct_inference(model, tokens, subject_pos, device, causal_layers, null_token, token_type, node_type, window_size, gauss=False):
    bsz = tokens.shape[0]

    patch_token_pos = get_patch_token_pos(tokens, subject_pos, device, token_type)
    
    tokens = tokens.repeat(len(causal_layers) + 2, 1)

    # inference: first is clean, last is corrupted
    result = model.run_with_hooks(
        tokens,
        fwd_hooks = [
            # pass stds instead of null token. If gauss, then "null_token" represent stds
            ("hook_embed", partial(gauss_subject_tokens, bsz, subject_pos, null_token) if gauss 
             else partial(replace_subject_tokens, bsz, subject_pos, null_token)),
            *[
                (partial(attn_out_filter if node_type == "attn" 
                        else mlp_out_filter, layer_no), 
                partial(patch_component_last_token, bsz, j, window_size) if token_type == "last"
                else partial(patch_component_token_pos, bsz, j, patch_token_pos, window_size)) 
                for j, layer_no in enumerate(causal_layers)
            ]
        ]
    )[:,-1].softmax(dim=-1)

    result = result.unflatten(0, (-1, bsz))

    # [batch, 1, d_vocab]
    target_result = result[0].unsqueeze(1)

    # [batch, n_layers, d_vocab]
    layer_results = result[1:].permute((1,0,2))

    target_loss, target_tokens = target_result.max(dim=-1)
    target_tokens = target_tokens.flatten()
    target_mask = torch.zeros_like(target_result).to(device)
    target_mask[torch.arange(target_result.shape[0]).to(device),0,target_tokens] = 1

    # to maximize AIE, minimize target-loss minus achieved-loss
    # [batch, 1]
    target_probs = target_loss
    # [batch, n_layers]
    layer_probs = (layer_results * target_mask).sum(dim=-1)

    return target_probs, layer_probs

def ct_inference_coherence(model, tokens, subject_pos, null_token, ablate_type,  n_samples_per_prompt=5, completion_tokens=10):
    bsz = tokens.shape[0]
    orig_length = tokens.shape[1]

    tokens = tokens.repeat(n_samples_per_prompt + 1, 1)

    for i in range(completion_tokens):
        result = model.run_with_hooks(
            tokens,
            fwd_hooks = [] if ablate_type is None else [
                # pass stds instead of null token. If gauss, then "null_token" represent stds
                ("hook_embed", partial(gauss_subject_tokens, bsz, subject_pos, null_token) if ablate_type == "gauss" 
                else partial(replace_subject_tokens, bsz, subject_pos, null_token)),
            ]
        )[:,-1].softmax(dim=-1)
        
        next_token = result.multinomial(num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=-1)
    
    # first one is unablated
    tokens = tokens.unflatten(0, (-1, bsz))[1:].flatten(start_dim=0, end_dim=1)
    return tokens[:, :orig_length], tokens[:, orig_length:]
    

        