import torch
from fancy_einsum import einsum
from functools import partial
import torch.optim
from pruners.Pruner import Pruner

class EdgePruner(Pruner):
    def __init__(self, *args, 
                 # when caching previous attention heads,
                 # cache the Z-value before applying the OV matrix
                 cache_compressed_attn=True, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.circs = ["q", "k", "v"]
        self.cache_compressed_attn = cache_compressed_attn

        if not self.cache_compressed_attn:
            self.base_model.cfg.use_attn_result = True

        self.post_bias = torch.stack([self.base_model.blocks[layer_no].attn.b_O.clone().detach() for layer_no in range(self.base_model.cfg.n_layers)], dim=0)

        if cache_compressed_attn:
            self.W_O = torch.stack([self.base_model.blocks[layer_no].attn.W_O.clone().detach() for layer_no in range(self.base_model.cfg.n_layers)], dim=0)
        
        self.cache_hooks = self.get_cache_hooks()
        self.patching_hooks = self.get_patching_hooks()

        self.attention_cache = []
        self.mlp_cache = []
        
        if self.counterfactual_mode:
            self.cf_attention_cache = []
            self.cf_mlp_cache = []
    
    def cache_hook_attention_all_tokens(self, activations, hook):
        if self.disable_hooks:
            return

        bsz = self.pruning_cfg.batch_size
        squeeze_activations = 0

        # if counterfactual mode, then first bsz are counterfact
        if self.counterfactual_mode:
            self.cf_attention_cache.append(activations[:bsz])
            squeeze_activations = bsz
        
        # if parallel inference, then next bsz are original
        if self.parallel_inference:
            squeeze_activations += bsz
        
        self.attention_cache.append(activations[squeeze_activations:])
        
        # need to return 1 input to continue inference, even if we don't need it
        return activations[:max(1, squeeze_activations)]
        
    def cache_hook_mlp_all_tokens(self, activations, hook):
        if self.disable_hooks:
            return

        bsz = self.pruning_cfg.batch_size
        squeeze_activations = 0

        # if counterfactual mode, then first bsz are counterfact
        if self.counterfactual_mode:
            self.cf_mlp_cache.append(activations[:bsz])
            squeeze_activations = bsz
        
        # if parallel inference, then next bsz are original
        if self.parallel_inference:
            squeeze_activations += bsz
        
        self.mlp_cache.append(activations[squeeze_activations:])

        # do inference on one example
        return activations[:max(1, squeeze_activations)]

    def retrieve_null_vals(self, attn_layer_no, mlp_layer_no):
        if self.counterfactual_mode:
            # bsz x seq_pos x i x n_heads x d_head
            if attn_layer_no > 0:
                null_attn = torch.stack(self.cf_attention_cache, dim=-3).detach()
            else:
                null_attn = None
            
            # bsz x seq_pos x i x d_model
            null_mlp = torch.stack(self.cf_mlp_cache, dim=-2).detach()

            if not self.condition_pos:
                for i, p in enumerate(self.perms):
                    if attn_layer_no > 0:
                        null_attn[i,:p.shape[0]] = null_attn[i,p]

                    null_mlp[i,:p.shape[0]] = null_mlp[i,p]

            if attn_layer_no > 0:
                null_attn = null_attn.repeat((self.pruning_cfg.n_samples, *[1 for _ in null_attn.shape[1:]]))
            
            null_mlp = null_mlp.repeat((self.pruning_cfg.n_samples, *[1 for _ in null_mlp.shape[1:]]))
        else:
            # (seq_pos x) i x n_heads x d_head
            null_attn = self.null_vals['attn'][...,:attn_layer_no,:,:]

            # (seq_pos x) i x d_model
            null_mlp = self.null_vals['mlp'][...,:mlp_layer_no,:]
            
        return null_attn, null_mlp

    # attention_constants: list of all constants for attention for layers thus far
    # mlp_constants: list of all constants for embed+mlp layers thus far
    # attention_cache: contains all attentions stored thus far, list of attention outputs by later
    # mlp_cache: list of mlp outputs by layer
    def pruning_edge_attention_hook_all_tokens(self, layer_no, circ_idx, orig_in, hook):
        if self.disable_hooks:
            return orig_in

        def prepend_orig(out):
            if self.parallel_inference or self.counterfactual_mode:
                out = torch.cat([orig_in, out], dim=0)

            # do not modify inference on BOS token
            out[:,[0]] = orig_in[0,[0]]
            return out
        
        # i is the current layer (0-indexed, equal to the number of layers before this one)
        # orig_in: batch x seq_pos x d_model
        # prune_mask[0]: (bsz * n_samples) x n_heads (dest) x i x n_heads (source)
        # attention_constants: i x n_heads (source) x d_model
        # attention_cache: i * [(bsz * n_samples) x seq_pos x n_heads (source) x d_model]

        # mlp_constants: (i+1) x d_model
        # mlp_cache: (i+1) * [(bsz * n_samples) x seq_pos x d_model]

        try:
            # one more for the input embedding
            # (bsz * n_samples x) (seq_pos x) i x n_heads (source) x d_head
            # (bsz * n_samples x) (seq_pos x) i x d_model
            null_attn, null_mlp = self.retrieve_null_vals(layer_no, layer_no+1)

            null_mlp = null_mlp.unsqueeze(-3)

            # mlp_mask: (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x 1 (d_model)
            mlp_mask = self.mask_sampler.sampled_mask["mlp-attn"][layer_no][:,circ_idx].unsqueeze(1).unsqueeze(-1)
            
            out = (mlp_mask * torch.stack(self.mlp_cache, dim=-2).unsqueeze(dim=2)).sum(dim=-2)
            out = out + (
                (mlp_mask < 0.001) * (1-mlp_mask) * null_mlp
                + (mlp_mask >= 0.001) * (1-mlp_mask) * null_mlp.detach()
            ).sum(dim=-2)

            if layer_no == 0:
                return prepend_orig(out)
            
            null_attn = null_attn.unsqueeze(-4)

            # attn_mask: (bsz * n_samples) x 1 (seq_pos) x n_heads (dest) x i x n_heads (source) x 1 (d_model/d_head)
            attn_mask = self.mask_sampler.sampled_mask["attn-attn"][layer_no][:,circ_idx].unsqueeze(1).unsqueeze(-1)
            attn_term = attn_mask * torch.stack(self.attention_cache, dim=-3).unsqueeze(dim=2) + (
                (attn_mask < 0.001) * (1-attn_mask) * null_attn
                + (attn_mask >= 0.001) * (1-attn_mask) * null_attn.detach()
            )

            # W_O: source_head x d_head x d_model
            if self.cache_compressed_attn:
                attn_term = einsum(
                            "batch pos dest_head prev_layer source_head d_head, \
                                prev_layer source_head d_head d_model -> \
                                batch pos dest_head d_model",
                            attn_term,
                            self.W_O[:layer_no]
                    )
            else:
                attn_term = attn_term.sum(dim=[-3,-2])
            out = out + attn_term + self.post_bias[:layer_no].sum(dim=0)

        except Exception as e:
            try:
                print(orig_in.shape)
                print(mlp_mask.shape)
                print(null_mlp.shape)
                print(attn_mask.shape)
                print(null_attn.shape)
                raise e
            except:
                raise e
        
        return prepend_orig(out)

    # same as attentions except not parallelized
    # attention_constants: list of all constants for attention for layers thus far
    # mlp_constants: list of all constants for embed+mlp layers thus far
    # attention_cache: contains all attentions stored thus far, list of attention outputs by later
    # mlp_cache: list of mlp outputs by layer
    def pruning_edge_mlp_hook_all_tokens(self, layer_no, orig_in, hook): 
        if self.disable_hooks:
            return orig_in
            
        def prepend_orig(out):
            if self.parallel_inference or self.counterfactual_mode:
                out = torch.cat([orig_in, out], dim=0)
            
            # do not modify inference on BOS token
            out[:,[0]] = orig_in[0,[0]]
            return out

        attn_layers_before = min(layer_no+1, self.base_model.cfg.n_layers)

        # i is the current layer (0-indexed, equal to the number of layers before this one)
        # orig_in: batch x seq_pos x d_model
        # prune_mask[0]: (bsz * n_samples) x i x n_heads
        # attention_constants: i x n_heads x d_model
        # attention_cache: i * [(bsz * n_samples) x seq_pos x n_heads x d_model]

        # mlp_constants: (i+1) x d_model
        # mlp_cache: (i+1) * [(bsz * n_samples) x seq_pos x d_model]

        try:
            # one more for the input embedding
            null_attn, null_mlp = self.retrieve_null_vals(attn_layers_before, layer_no+1)

            # (bsz * n_samples) x 1 (seq_pos) x i x 1 (d_model)
            mlp_mask = self.mask_sampler.sampled_mask["mlp-mlp"][layer_no].unsqueeze(1).unsqueeze(-1)

            out = (mlp_mask * torch.stack(self.mlp_cache, dim=2)).sum(dim=2)

            out = out + (
                (mlp_mask < 0.001) * (1-mlp_mask) * null_mlp
                + (mlp_mask >= 0.001) * (1-mlp_mask) * null_mlp.detach()
            ).sum(dim=2)
            
            # (bsz * n_samples) x 1 (seq_pos) x i x n_heads x 1 (d_model or d_head)
            attn_mask = self.mask_sampler.sampled_mask["attn-mlp"][layer_no].unsqueeze(1).unsqueeze(-1)
            attn_term = attn_mask * torch.stack(self.attention_cache, dim=-3) + (
                (attn_mask < 0.001) * (1-attn_mask) * null_attn
                + (attn_mask >= 0.001) * (1-attn_mask) * null_attn.detach()
            )

            # W_O: source_head x d_head x d_model
            if self.cache_compressed_attn:
                attn_term = einsum(
                            "batch pos prev_layer source_head d_head, \
                                prev_layer source_head d_head d_model -> \
                                batch pos d_model",
                            attn_term,
                            self.W_O[:attn_layers_before]
                    )
            else:
                attn_term = attn_term.sum(dim=[-3,-2])

            out = out + attn_term + self.post_bias[:attn_layers_before].sum(dim=0)
        except Exception as e:
            try:
                print(orig_in.shape)
                print(mlp_mask.shape)
                print(null_mlp.shape)
                print(attn_mask.shape)
                print(null_attn.shape)
                raise e
            except:
                raise e
        
        return prepend_orig(out)

    def pruning_edge_final_hook_all_tokens(self, orig_in, hook):
        out = self.pruning_edge_mlp_hook_all_tokens(self.base_model.cfg.n_layers, orig_in, hook)
        if self.disable_hooks:
            out = out.unsqueeze(0)
        else:
            # CONVENTION: since we repeat the batch tokens n_samples times, the correct unflattened shape for embeddings is [n_samples, batch_size, seq, d_model]
            out = out.unflatten(0, (-1, self.pruning_cfg.batch_size))
            if self.counterfactual_mode:
                out = out[1:]
        out = (out * self.last_token_mask.unsqueeze(-1)).sum(dim=2)
        return out

    def get_cache_hooks(self):
        embed_filter = lambda name: name == f"blocks.{0}.hook_resid_pre"
        attention_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_result"
        attention_compressed_filter = lambda layer_no, name: name == f"blocks.{layer_no}.attn.hook_z"
        mlp_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_out"

        n_layers = self.base_model.cfg.n_layers

        return {
            # cache embedding
            "cache_embed": (embed_filter, 
                            self.cache_hook_mlp_all_tokens),

            # cache attention (at z if compressed)
            **{
                f"cache_attn_{layer_no}": 
                (partial(attention_compressed_filter 
                         if self.cache_compressed_attn 
                         else attention_points_filter, layer_no), 
                self.cache_hook_attention_all_tokens) 
                for layer_no in range(n_layers)
            },

            # cache MLP
            **{
                f"cache_mlp_{layer_no}": (partial(mlp_points_filter, layer_no), 
                                          self.cache_hook_mlp_all_tokens)
                for layer_no in range(n_layers)
            }
        }
    
    def get_patching_hooks(self):
        attention_in_filter = lambda layer_no, circ, name: name == f"blocks.{layer_no}.hook_{circ}_input"
        mlp_in_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_mlp_in"
        final_embed_filter = lambda name: name == f"blocks.{n_layers-1}.hook_resid_post"

        n_layers = self.base_model.cfg.n_layers
        
        return {
            # patch attention (recompute O-matrix if compressed)
            **{
                f"patch_attn_{layer_no}_{circ}": 
                (partial(attention_in_filter, layer_no, circ), 
                # partial(self.time_hook, f"attn_{layer_no}", 
                partial(self.pruning_edge_attention_hook_all_tokens, 
                        layer_no, j)) 
                for layer_no in range(n_layers) 
                for j, circ in enumerate(self.circs)
            },

            # patch MLP (recompute O-matrix if compressed)
            **{
                f"patch_mlp_{layer_no}": 
                (partial(mlp_in_filter, layer_no), 
                # partial(self.time_hook, f"mlp_{layer_no}", 
                partial(self.pruning_edge_mlp_hook_all_tokens, 
                        layer_no)) 
                for layer_no in range(n_layers)
            },

            # patch MLP (recompute O-matrix if compressed)
            "patch_final": 
            (final_embed_filter, 
                # partial(self.time_hook, f"output_embeds", 
                        self.pruning_edge_final_hook_all_tokens)
                # )
        }
    
    def setup_inference(self, batch, last_token_pos):
        self.attention_cache.clear()
        self.mlp_cache.clear()

        if self.counterfactual_mode:
            self.cf_attention_cache.clear()
            self.cf_mlp_cache.clear()

        super().setup_inference(batch, last_token_pos)