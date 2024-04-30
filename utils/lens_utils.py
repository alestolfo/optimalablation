# %%
import torch
from fancy_einsum import einsum

# %%
def apply_lens(model, lens_weights, lens_bias, activation_storage):
    if not isinstance(lens_weights, torch.Tensor):
        lens_weights = torch.stack(lens_weights, dim=0)
    linear_lens_output = einsum("layer result activation, layer batch activation -> batch layer result", lens_weights, torch.stack(activation_storage, dim=0)) + torch.stack(lens_bias, dim=0)
    linear_lens_output = model.ln_final(linear_lens_output)
    linear_lens_probs = model.unembed(linear_lens_output).softmax(dim=-1)
    return linear_lens_probs

# %%
# shared_bias: optimal constants are shared between modal layers. 
# if shared_bias is false, then attn_bias is a list where elt i has shape [i, d_model]
# if shared_biias is true, then attn_bias is a list where elt i has shape [1, d_model]
def apply_modal_lens(model, attn_bias, activation_storage, shared_bias=False):
    resid = []
    for layer_no in range(model.cfg.n_layers):
        if layer_no > 0:
            # [layer_no, batch, d_model]
            resid = torch.cat([resid_mid,activation_storage[layer_no].unsqueeze(0)], dim=0)
        else:
            resid = activation_storage[layer_no].unsqueeze(0)
        if shared_bias:
            attn_bias_layer = attn_bias[layer_no].unsqueeze(0)
        else:
            attn_bias_layer = attn_bias[layer_no]
        resid_mid = resid + attn_bias_layer.unsqueeze(1)
        normalized_resid_mid = model.blocks[layer_no].ln2(resid_mid)
        mlp_out = model.blocks[layer_no].mlp(normalized_resid_mid)
        resid = resid_mid + mlp_out
    
    # [n_layers, batch, d_model]
    resid = model.ln_final(resid)

    # [batch, n_layers, d_vocab]
    modal_lens_probs = model.unembed(resid).softmax(dim=-1).permute((1,0,2))
    return modal_lens_probs

# %%
def apply_lmlp_lens(model, attn_bias, activation_storage, n_layers, shared_bias=False):
    resid = torch.stack(activation_storage, dim=0)

    if shared_bias:
        attn_bias = attn_bias.unsqueeze(0)
    resid_mid = resid + attn_bias.unsqueeze(1)
    normalized_resid_mid = model.blocks[n_layers - 1].ln2(resid_mid)
    mlp_out = model.blocks[n_layers - 1].mlp(normalized_resid_mid)
    resid = resid_mid + mlp_out

    # [n_layers, batch, d_model]
    resid = model.ln_final(resid)

    # [batch, n_layers, d_vocab]
    modal_lens_probs = model.unembed(resid).softmax(dim=-1).permute((1,0,2))
    return modal_lens_probs
