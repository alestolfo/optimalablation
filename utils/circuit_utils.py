# %%
import torch
import os
import numpy as np 
import torch.optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
n_layers = 12
n_heads = 12
device="cuda:0"

layers_to_prune = []
for layer_no in range(n_layers):
    layers_to_prune.append(("attn", layer_no))
    layers_to_prune.append(("mlp", layer_no))
layers_to_prune.append(("mlp", n_layers))

edge_prune_mask = {
    "attn-attn": [
        # edges from attn layers into q, k, v circuits
        # first n_heads is the receiver dimension
        torch.ones((1, 3, n_heads, i, n_heads)).to(device)
        for i in range(n_layers)
    ], 
    "mlp-attn": [
        # edges from input embedding and previous MLP to q, k, v circuits
        torch.ones((1, 3, n_heads, i+1)).to(device)
        for i in range(n_layers)
    ],
    "attn-mlp": [
        # edges from attn layers into MLP and output embedding
        torch.ones((1, min(i+1, n_layers), n_heads)).to(device)
        for i in range(n_layers+1)
    ],
    "mlp-mlp": [
        # edges from input embedding and previous MLP to MLP and output embedding
        torch.ones((1, i+1)).to(device)
        for i in range(n_layers + 1)
    ]
}

vertex_prune_mask = {
    "attn": [
        torch.ones((1, n_heads)).to(device) 
        for _ in range(n_layers)
    ],
    "mlp": [
        torch.ones((1,)).to(device)
        for _ in range(n_layers)
    ]
}

IOI_MANUAL_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}

def get_ioi_nodes(return_tensor=False):
    nodes = {"attn": [], "mlp": []}
    for k in IOI_MANUAL_CIRCUIT:
        for node in IOI_MANUAL_CIRCUIT[k]:
            nodes['attn'].append([node[0], node[1]])
    if return_tensor: 
        nodes['attn'] = torch.tensor(nodes['attn']).to(device)
    return nodes

def get_ioi_edge_mask():
    init_mask = nodes_to_mask(get_ioi_nodes())

    embed_in_edges = {
        0: ['s2 inhibition', 'duplicate token', 'previous token'],
        1: ['name mover', 'backup name mover', 'negative', 'previous token', 'duplicate token']
    }

    for layer_mask in init_mask['mlp-attn']:
        layer_mask[...,0] = 0

    for head_type in embed_in_edges[0]:
        for layer_idx, head_idx in IOI_MANUAL_CIRCUIT[head_type]:
            init_mask['mlp-attn'][layer_idx][:,0,head_idx,0] = 1
    
    for head_type in embed_in_edges[1]:
        for layer_idx, head_idx in IOI_MANUAL_CIRCUIT[head_type]:
            init_mask['mlp-attn'][layer_idx][:,1:,head_idx,0] = 1

    embed_out_edges = ['name mover', 'backup name mover', 'negative']

    init_mask['attn-mlp'][-1] *= 0

    for layer_mask in init_mask['attn-attn']:
        layer_mask *= 0

    for head_type in embed_out_edges:
        for layer_idx, head_idx in IOI_MANUAL_CIRCUIT[head_type]:
            init_mask['attn-mlp'][-1][:,layer_idx,head_idx] = 1
            for src_layer, src_head in IOI_MANUAL_CIRCUIT['s2 inhibition']:
                init_mask['attn-attn'][layer_idx][:,0,head_idx,src_layer, src_head] = 1
    
    for layer_idx, head_idx in IOI_MANUAL_CIRCUIT['s2 inhibition']:
        for head_type in ['duplicate token', 'induction']:
            for src_layer, src_head in IOI_MANUAL_CIRCUIT[head_type]:
                init_mask['attn-attn'][layer_idx][:,1:,head_idx,src_layer, src_head] = 1

    for layer_idx, head_idx in IOI_MANUAL_CIRCUIT['induction']:
        for src_layer, src_head in IOI_MANUAL_CIRCUIT['previous token']:
            init_mask['attn-attn'][layer_idx][:,1:,head_idx,src_layer, src_head] = 1
    
    init_mask = edges_to_mask(torch.load("acdc_ioi_runs/edges_manual.pth"))
    return init_mask            

# %%
def retrieve_mask(folder, state_dict=False):
    snapshot_path = f"{folder}/snapshot.pth"

    if os.path.exists(snapshot_path):
        print("Loading previous training run")
        previous_state = torch.load(snapshot_path)

        prune_mask = {}
        for k in previous_state['pruner_dict']:
            if not k.startswith("mask_sampler"):
                continue
            s = k.split(".")
            if s[-2] not in prune_mask:
                prune_mask[s[-2]] = []
            prune_mask[s[-2]].append(previous_state['pruner_dict'][k][...,0].unsqueeze(0))
            if int(s[-1])+1 != len(prune_mask[s[-2]]):
                print("WARNING: out of order")
    else:
        print("TRAINING RUN NOT FOUND")
        return None, None
    
    if state_dict:
        return prune_mask, previous_state['pruner_dict']
    else:
        return prune_mask

# %%

def plot_mask(prune_mask):
    all_alphas = torch.cat([ts.flatten() for k in prune_mask for ts in prune_mask[k]], dim=0)
    sorted_values, _ = torch.sort(all_alphas)
    sns.histplot(sorted_values.cpu())

# %%
    
def total_edges(mask):
    return np.sum([np.sum([torch.sum(ts > 0).item() for ts in mask[k]]) for k in mask])
    
def discretize_mask(prune_mask, threshold):
    filtered_prune_mask = {k: [
        (ts > threshold) * 1 for ts in prune_mask[k]
    ] for k in prune_mask}
    return filtered_prune_mask

# def apply_mask_inference(mask):
#     return {k: [ts.unsqueeze(0) for ts in mask[k]] for k in mask}

# attn-attn: (bsz * n_samples) x n_heads (dest) x i x n_heads (source)
# mlp-attn: (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x 1 (d_model)

# attn-mlp: (bsz * n_samples) x i x n_heads
# mlp-mlp: (bsz * n_samples) x 1 (seq_pos) x i x 1 (d_model)

# check for dangling edges
def prune_dangling_edges(filtered_prune_mask, bsz=1, skip_filtering=False, node_counts=False):
    if bsz == 1:
        num_edges = total_edges(filtered_prune_mask)
        print("num edges", num_edges)

    if skip_filtering:
        return filtered_prune_mask, num_edges, num_edges, None, None

    attn_edges_out = torch.zeros((bsz, n_layers, n_heads)).to(device)
    mlp_edges_out = torch.zeros((bsz, n_layers + 2)).to(device)
    mlp_edges_out[:,-1] = 1

    for component_type, cur_layer in reversed(layers_to_prune):
        if component_type == "mlp":
            # bsz, 1
            this_layer_mlp = (mlp_edges_out[:, cur_layer+1] > 0).unsqueeze(-1)

            # bsz, prev_layer, prev_head
            filtered_prune_mask['attn-mlp'][cur_layer] *= this_layer_mlp.unsqueeze(-1)
            attn_edges_out[:, :min(cur_layer+1, n_layers)] += filtered_prune_mask['attn-mlp'][cur_layer] * this_layer_mlp.unsqueeze(-1)

            # bsz, prev_layer
            filtered_prune_mask['mlp-mlp'][cur_layer] *= this_layer_mlp
            mlp_edges_out[:, :cur_layer+1] += filtered_prune_mask['mlp-mlp'][cur_layer] * this_layer_mlp

        if component_type == "attn":
            # bsz, 1, cur_head, 1
            this_layer_heads = (attn_edges_out[:, [cur_layer]] > 0).unsqueeze(-1)

            # bsz, circ, cur_head, prev_layer, prev_head
            filtered_prune_mask['attn-attn'][cur_layer] *= this_layer_heads.unsqueeze(-1)
            attn_edges_out[:, :cur_layer] += (filtered_prune_mask['attn-attn'][cur_layer] * this_layer_heads.unsqueeze(-1)).sum(dim=[1,2])

            # bsz, circ, cur_head, prev_layer
            filtered_prune_mask['mlp-attn'][cur_layer] *= this_layer_heads
            mlp_edges_out[:, :cur_layer+1] += (filtered_prune_mask['mlp-attn'][cur_layer] * this_layer_heads).sum(dim=[1,2])
        
    attn_edges_in = torch.zeros((bsz, n_layers, n_heads)).to(device)
    mlp_edges_in = torch.zeros((bsz, n_layers + 2)).to(device)
    mlp_edges_in[:,0] = 1

    for component_type, cur_layer in layers_to_prune:
        if component_type == "mlp":
            # bsz, prev_layer, prev_head
            filtered_prune_mask['attn-mlp'][cur_layer] *= (attn_edges_in[:, :min(cur_layer+1, n_layers)] > 0)
            mlp_edges_in[:, cur_layer+1] += filtered_prune_mask['attn-mlp'][cur_layer].sum()
            
            # bsz, prev_layer
            filtered_prune_mask['mlp-mlp'][cur_layer] *= (mlp_edges_in[:, :cur_layer+1] > 0)     
            mlp_edges_in[:, cur_layer+1] += filtered_prune_mask['mlp-mlp'][cur_layer].sum()

        if component_type == "attn":
            # bsz, circ, cur_head, prev_layer, prev_head
            filtered_prune_mask['attn-attn'][cur_layer] *= (attn_edges_in[:, :cur_layer] > 0).unsqueeze(1).unsqueeze(2)
            attn_edges_in[:, cur_layer] += filtered_prune_mask['attn-attn'][cur_layer].sum(dim=[1,-2,-1])

            # bsz, circ, cur_head, prev_layer
            filtered_prune_mask['mlp-attn'][cur_layer] *= (mlp_edges_in[:, :cur_layer+1] > 0).unsqueeze(1).unsqueeze(2)
            attn_edges_in[:, cur_layer] += filtered_prune_mask['mlp-attn'][cur_layer].sum(dim=[1,-1])

    if bsz == 1:
        clipped_num_edges = total_edges(filtered_prune_mask)
        print("num edges after dangling edges removed", clipped_num_edges)
        
        if node_counts:
            return filtered_prune_mask, num_edges, clipped_num_edges, attn_edges_in + attn_edges_out, mlp_edges_in[:,1:-1] + mlp_edges_out[:,1:-1]
        else:
            return filtered_prune_mask, num_edges, clipped_num_edges, attn_edges_in, mlp_edges_in[:,1:-1]

    return filtered_prune_mask

# %%

def clone_constant_mask(mask=edge_prune_mask):
    prune_mask = {}
    for k in mask:
        prune_mask[k] = []
        for ts in mask[k]:
            prune_mask[k].append(ts.clone())
    return prune_mask

# %%
# NOTE: MLPs are recorded 0 to n_layers + 1 (0-13)
def mask_to_edges(mask):
    all_edges = {}
    edge_count = 0
    for k in mask:
        type_edges = []
        for i,ts in enumerate(mask[k]):
            # edge_count x len(ts.shape)
            edges = ts.nonzero()
            if edges.nelement() == 0:
                continue
            edges = torch.cat([torch.ones((edges.shape[0],1)).to(device) * i, edges[:, 1:]], dim=1)
            if k.endswith("attn"):
                edges[:,:2] = edges[:,[1,0]]
            else:
                edges[:,0] += 1
            edge_count += len(edges)
            type_edges.append(edges)
        all_edges[k] = torch.cat(type_edges, dim=0)
    return all_edges, edge_count

def mask_diff(large_mask, small_mask):
    all_edges = {}
    for k in large_mask:
        type_edges = []
        for i, ts in enumerate(large_mask[k]):
            type_edges.append((ts - small_mask[k][i]).nonzero())
        all_edges[k] = type_edges
    return all_edges

# returns mlps 0-11
def mask_to_nodes(mask, mask_type="edges", return_tensor=False):
    if mask_type=="edges":
        _, _, _, attn_nodes, mlp_nodes = prune_dangling_edges(mask)
    else:
        attn_nodes = torch.stack(mask['attn'], dim=1)
        mlp_nodes = torch.stack(mask['mlp'], dim=1)        
    attn_nodes = attn_nodes.squeeze(0).nonzero()
    mlp_nodes = mlp_nodes.squeeze(0).nonzero()
    node_count = attn_nodes.shape[0] + mlp_nodes.shape[0]
    if return_tensor:
        return {"attn": attn_nodes, "mlp": mlp_nodes}, node_count
    return {"attn": attn_nodes.cpu().numpy().tolist(), "mlp": mlp_nodes.flatten().cpu().numpy().tolist()}, node_count

# NOTE: when edges are passed in it's expected that MLPs are record 0 to 13. 0 input, 1-12 mlps, 13 output.
# this function is really bad
def edges_to_mask(edges):
    prune_mask = {}
    for k in edge_prune_mask:
        prune_mask[k] = []
        if edges[k].nelement() > 0:
            if k.endswith("attn"):
                edges[k][:,:2] = edges[k][:,[1,0]]
            else:
                edges[k][:,0] -= 1
        edges_k = torch.cat([torch.zeros((edges[k].shape[0],1)).to(device), edges[k].to(device)], dim=1)
        for i, ts in enumerate(edge_prune_mask[k]):
            this_layer_mask = ts.clone() * 0
            if edges_k.nelement() > 0:
                edge_idxs = (edges_k[:,1] == i).nonzero()
                if edge_idxs.nelement() > 0:
                    this_layer_edges = edges_k[edge_idxs[:,0]].int()
                    this_layer_slice = tuple([this_layer_edges[:,0], *[this_layer_edges[:,i] for i in range(2, this_layer_edges.shape[1])]])
                    this_layer_mask[this_layer_slice] = 1
            prune_mask[k].append(this_layer_mask)
    return prune_mask

# expect MLPs 0-11
def nodes_to_mask(nodes, all_mlps=True):
    excluded_heads = set([(layer_idx, head_idx) for layer_idx in range(n_layers) for head_idx in range(n_heads)])

    prune_mask = clone_constant_mask()

    for layer_idx, head_idx in nodes['attn']:
        excluded_heads.remove((layer_idx,head_idx))

    for layer_idx, head_idx in list(excluded_heads):
        prune_mask["attn-attn"][layer_idx][:,:,head_idx] = 0
        prune_mask["mlp-attn"][layer_idx][:,:,head_idx] = 0
        for future_layer in range(layer_idx+1,n_layers):
            prune_mask["attn-attn"][future_layer][:,:,:,layer_idx,head_idx] = 0
        for future_layer in range(layer_idx,n_layers+1):
            prune_mask["attn-mlp"][future_layer][:,layer_idx,head_idx] = 0
    
    if all_mlps:
        return prune_mask
    
    excluded_mlps = set([layer_idx for layer_idx in range(n_layers)])
    
    for layer_idx in nodes['mlp']:
        excluded_mlps.remove(layer_idx)

    for layer_idx in list(excluded_mlps):
        prune_mask["attn-mlp"][layer_idx] *= 0
        prune_mask["mlp-mlp"][layer_idx] *= 0
        for future_layer in range(layer_idx+1,n_layers):
            prune_mask["mlp-attn"][future_layer][:,:,:,layer_idx+1] = 0
        for future_layer in range(layer_idx+1,n_layers+1):
            prune_mask["mlp-mlp"][future_layer][:,layer_idx+1] = 0
    
    return prune_mask

# expect MLPs 0-11
def nodes_to_vertex_mask(nodes, all_mlps=True):

    included_heads = set([(x[0], x[1]) for x in nodes['attn']])

    prune_mask = clone_constant_mask(vertex_prune_mask)

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            if (layer_idx, head_idx) not in included_heads:
                prune_mask['attn'][layer_idx][:, head_idx] = 0

    if all_mlps:
        return prune_mask
    
    node_mlps = set(nodes['mlp'])
    
    for layer_idx in range(n_layers):
        if layer_idx not in node_mlps:
            prune_mask['mlp'][layer_idx] *= 0
    
    return prune_mask

# %%
