        
all_masks = {}

manual_mask, e, c_e, manual_attn_in, manual_mlp_in = get_mask(constant_prune_mask, 0.5)

for reg_lamb in [5e-3, 2e-3, 1e-3, 7e-4, 5e-4, 2e-4]:
    for cutoff in [1.5,0,-1]:
        prune_mask = retrieve_mask(reg_lamb)
        cpm, _, clipped_edges, attn_in, mlp_in = get_mask(prune_mask, cutoff)
        all_masks[str(reg_lamb) + " " + str(cutoff)] = (clipped_edges, cpm, attn_in, mlp_in)

all_masks["manual"] = (c_e, manual_mask, manual_attn_in, manual_mlp_in)
