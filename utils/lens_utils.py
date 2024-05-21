# %%
import torch
from fancy_einsum import einsum
import os
from utils.training_utils import save_hook_last_token, save_hook_last_token_bsz, resid_points_filter, plot_no_outliers, gen_resample_perm, attn_out_filter
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pickle
from functools import partial

# %%
kl_loss = torch.nn.KLDivLoss(reduction="none")

# v1: [b, l, d_vocab], v2: [b, l, d_vocab], orig_dist: [b, 1, d_vocab]
def a_inner_prod(v1, v2, orig_dist):
    assert (orig_dist.sum(dim=-1) - 1).abs().mean() <= 1e-3
    geom_mean_1 = (orig_dist * v1.log()).sum(dim=-1, keepdim=True).exp()
    geom_mean_2 = (orig_dist * v2.log()).sum(dim=-1, keepdim=True).exp()

    return (orig_dist * (v1 / geom_mean_1).log() * (v2 / geom_mean_2).log()).sum(dim=-1)

def a_sim(v1, v2, orig_dist):
    return a_inner_prod(v1, v2, orig_dist) / (a_inner_prod(v1, v1, orig_dist) * (a_inner_prod(v2, v2, orig_dist))).sqrt()

def compile_loss_dfs(all_losses, lens_losses_dfs, suffix=""):
    for k in all_losses:
        if not isinstance(all_losses[k], torch.Tensor):
            if len(all_losses[k]) == 0:
                continue
            all_losses[k] = torch.cat(all_losses[k], dim=0)
        lens_loss_df = pd.DataFrame(all_losses[k].cpu().numpy())
        lens_loss_df.columns = [f"{k}_{x}" for x in lens_loss_df.columns]
        lens_losses_dfs[f"{k}{suffix}"] = lens_loss_df
    return lens_losses_dfs

# plotting tuned vs OCA lens performance
def corr_plot(lens_loss_1, lens_loss_2, key_1, key_2, n_layers, offset=0):
    tuned_modal_comp = lens_loss_1.merge(lens_loss_2, left_index=True, right_index=True)

    f, axes = plt.subplots((n_layers-offset-1)//3 + 1, 3, figsize=(15,15))
    f2, axes_log = plt.subplots((n_layers-offset-1)//3 + 1, 3, figsize=(15,15))

    # correlation plot
    for i in range(offset,n_layers):
        a_idx = i - offset
        plot_no_outliers(
            sns.histplot, .01, 
            tuned_modal_comp[f"{key_1}_{i}"], tuned_modal_comp[f"{key_2}_{i}"], 
            axes[a_idx // 3, a_idx % 3], xy_line=True, 
            args={"x": f"{key_1}_{i}", "y": f"{key_1}_{i}"})
        
        plot_no_outliers(
            sns.histplot, 0,
            np.log(tuned_modal_comp[f"{key_1}_{i}"]), np.log(tuned_modal_comp[f"{key_2}_{i}"]),
            axes_log[a_idx // 3, a_idx % 3], xy_line=True,
            args={"x": f"{key_1}_{i}", "y": f"{key_2}_{i}"}
        )
    f.show()
    f2.show()
    plt.close(f)
    plt.close(f2)

def overall_comp(lens_losses_dfs, title="Lens losses", save=None, offset=0):
    # overall comparison line plot
    lens_loss_means = {}
    for k in lens_losses_dfs:
        lens_loss_means[k] = lens_losses_dfs[k].mean()
        w = 0.5 if k.startswith("shrinkage") else None
        label = None if k.startswith("shrinkage") else k
        ax = sns.lineplot(x=np.arange(len(lens_loss_means[k]) - offset) + offset, y=lens_loss_means[k][offset:], label=label, linewidth=w)
        ax.set(xlabel="layer", ylabel="KL-divergence", title=title)
    if save:
        plt.savefig(save)
    plt.show()
    plt.close()

class LensExperiment():
    def __init__(self, model, owt_iter, folders, device, shared_bias=False, pretrained=True):
        self.model = model
        self.owt_iter = owt_iter
        self.folders = folders
        self.device = device

        # for modal lens
        self.shared_bias = shared_bias

        self.all_lens_weights, self.all_lens_bias = self.load_lens(folders)

        if pretrained:
            n_layers = model.cfg.n_layers

            self.act_means, self.a_mtrx = self.compute_act_dist(folders['linear_oa'])
            self.std_intvs = np.logspace(-1.5, 0.2, 50)
            self.std_mult = (math.log(1.05) * (n_layers - 1 - torch.arange(n_layers).to(device))).exp()
            self.perturb_mag = torch.tensor((self.std_intvs[:,None] * self.std_mult.cpu().numpy()).T, device=device)
            self.perturb_losses = self.get_perturb_magnitudes(folders['linear_oa'])
        
        # for layer_no in range(n_layers):
        #     sns.lineplot(x=self.perturb_mag[layer_no].cpu(), y=self.perturb_losses[layer_no].cpu(), label=layer_no)

    def load_lens(self, folders):
        all_lens_weights = {}
        all_lens_bias = {}

        for k in folders:
            if os.path.exists(f"{folders[k]}/lens_weights.pkl"):
                with open(f"{folders[k]}/lens_weights.pkl", "rb") as f:
                    all_lens_weights[k] = pickle.load(f)
            if os.path.exists(f"{folders[k]}/lens_bias.pkl"):
                with open(f"{folders[k]}/lens_bias.pkl", "rb") as f:
                    all_lens_bias[k] = pickle.load(f)
            if k == "mean":
                path = f"{folders[k]}/attn_means.pth"
                if os.path.exists(path):
                    means = torch.load(path)
                else:
                    means = self.collect_attn_means(folders[k])
                all_lens_bias["mean"] = means
        
        return all_lens_weights, all_lens_bias
    
    def collect_attn_means(self, folder):
        n_layers = self.model.cfg.n_layers
        path = f"{folder}/attn_means.pth"

        if os.path.exists(path):
            means = torch.load(path)
        else:
            all_activations = []
            for i in tqdm(range(10000)):
                batch = next(self.owt_iter)['tokens']

                with torch.no_grad():
                    activation_storage = []

                    target_probs = self.model.run_with_hooks(
                        batch,
                        fwd_hooks=[
                                *[(partial(attn_out_filter, layer_no), 
                                partial(save_hook_last_token, activation_storage)) 
                                for layer_no in range(n_layers)],
                            ]
                    )[:,-1].softmax(dim=-1).unsqueeze(1)

                    # batch, d_model, n_layers
                    all_activations.append(torch.stack(activation_storage, dim=-1).mean(dim=0, keepdim=True))
            
            all_activations = torch.cat(all_activations, dim=0)

            act_means = []
            # all_activations: [samples, d_model, n_layers]
            for j in range(n_layers):
                act_means.append(all_activations[...,j].mean(dim=0))

            torch.save(act_means, path)
        return act_means
    
    def compute_act_dist(self, folder):
        n_layers = self.model.cfg.n_layers

        # we need to compute mean activations to analyze projection
        # compute std/cov for strength of ``local'' causal perturbations
        if (os.path.exists(f"{folder}/act_means.pth") 
            and os.path.exists(f"{folder}/covs.pth")):
            act_means = torch.load(f"{folder}/act_means.pth")
            act_covs = torch.load(f"{folder}/covs.pth")
            # stds = []
            # for j in range(n_layers):
            #     stds.append(act_covs[j].diag().sqrt())
        else:
            # compute act means
            # act_means = [0 for _ in range(n_layers)]
            # stds = [torch.zeros(d_model).to(device) for _ in range(n_layers)]

            all_activations = []
            for i in tqdm(range(10000)):
                batch = next(self.owt_iter)['tokens']

                with torch.no_grad():
                    activation_storage = []

                    target_probs = self.model.run_with_hooks(
                        batch,
                        fwd_hooks=[
                                *[(partial(resid_points_filter, layer_no), 
                                partial(save_hook_last_token, activation_storage)) 
                                for layer_no in range(n_layers)],
                            ]
                    )[:,-1].softmax(dim=-1).unsqueeze(1)

                    all_activations.append(torch.stack(activation_storage, dim=-1))
            
            all_activations = torch.cat(all_activations, dim=0)

            act_means = []
            act_covs = []
            # all_activations: [samples, d_model, n_layers]
            for j in range(n_layers):
                act_means.append(all_activations[...,j].mean(dim=0))

                # cov takes rows=variables, columns=observations
                act_covs.append(all_activations[...,j].permute((1,0)).cov())

            torch.save(act_means, f"{folder}/act_means.pth")
            torch.save(act_covs, f"{folder}/covs.pth")

        # [n_layers, d_mvn, d_model], d_mvn = d_model
        a_mtrx = torch.stack([torch.linalg.cholesky(covs + 3e-4 * torch.eye(covs.shape[0]).to(self.device)) for covs in act_covs], dim=0)

        assert a_mtrx.isnan().sum() == 0

        # fix weird bug with cholesky where the last entry is nan
        # for j in range(a_mtrx.shape[0]):
        #     a_mtrx[j,-1,-1] = (act_covs[j][-1,-1] - a_mtrx[j,-1,:-1].square().sum()).sqrt()
        # diff_mtrx = a_mtrx[0] @ a_mtrx[0].permute((1,0)) - act_covs[0]
        # assert (diff_mtrx.abs() > 1e-6).sum() == 0

        return act_means, a_mtrx

    def apply_lens(self, lens_name, activation_storage):
        lens_weights = self.all_lens_weights[lens_name]
        lens_bias = self.all_lens_bias[lens_name]

        if not isinstance(lens_weights, torch.Tensor):
            lens_weights = torch.stack(lens_weights, dim=0)
        if not isinstance(lens_bias, torch.Tensor):
            lens_bias = torch.stack(lens_bias, dim=0)
        
        linear_lens_output = einsum("layer result activation, layer batch activation -> batch layer result", lens_weights, torch.stack(activation_storage, dim=0)) + lens_bias
        linear_lens_output = self.model.ln_final(linear_lens_output)
        linear_lens_probs = self.model.unembed(linear_lens_output).log_softmax(dim=-1)
        return linear_lens_probs

    # returns LOGGED probs
    def apply_modal_lens(self, lens_name, activation_storage, shared_bias=False, attention_storage=None):
        if lens_name == "resample":
            permutation = gen_resample_perm(activation_storage[0].shape[0]).to(self.device)
        else:
            attn_bias = self.all_lens_bias[lens_name]

        resid = []
        for layer_no in range(self.model.cfg.n_layers):
            if layer_no > 0:
                # [layer_no, batch, d_model]
                resid = torch.cat([resid,activation_storage[layer_no].unsqueeze(0)], dim=0)
            else:
                resid = activation_storage[layer_no].unsqueeze(0)
            # no shared_bias: [layer_no+1, d_model]

            if lens_name == "resample":
                resid_mid = resid + attention_storage[layer_no][permutation]
            else:
                attn_bias_layer = attn_bias[layer_no]
                if shared_bias:
                    # shared_bias: [d_model,]
                    attn_bias_layer = attn_bias_layer.unsqueeze(0)

                resid_mid = resid + attn_bias_layer.unsqueeze(1)
            
            normalized_resid_mid = self.model.blocks[layer_no].ln2(resid_mid)
            mlp_out = self.model.blocks[layer_no].mlp(normalized_resid_mid)
            resid = resid_mid + mlp_out
        
        # [n_layers, batch, d_model]
        resid = self.model.ln_final(resid)

        modal_lens_probs = self.model.unembed(resid)

        # [batch, n_layers, d_vocab]
        modal_lens_probs = modal_lens_probs.log_softmax(dim=-1).permute((1,0,2))
        return modal_lens_probs

    # returns LOGGED probs
    def apply_lmlp_lens(self, attn_bias, activation_storage, shared_bias=False):
        n_layers = self.model.cfg.n_layers
        
        resid = torch.stack(activation_storage, dim=0)

        if shared_bias:
            attn_bias = attn_bias.unsqueeze(0)
        resid_mid = resid + attn_bias.unsqueeze(1)
        normalized_resid_mid = self.model.blocks[n_layers - 1].ln2(resid_mid)
        mlp_out = self.model.blocks[n_layers - 1].mlp(normalized_resid_mid)
        resid = resid_mid + mlp_out

        # [n_layers, batch, d_model]
        resid = self.model.ln_final(resid)

        # [batch, n_layers, d_vocab]
        modal_lens_probs = self.model.unembed(resid).log_softmax(dim=-1).permute((1,0,2))
        return modal_lens_probs

    # std: scalar or shape [n_layers, d_model]
    # fixed_dir: "perturb" (local perturbation), "steer" (add a vector), "project" (linear projection out of a direction), False
    # if fixed_dir is perturb, then std is should gives us the Cholesky matrix multiplied by a scalar that gives the overall scale (stdevs from mean).
    # if fixed_dir is steer, then std gives the direction to add 
    # if fixed_dir is project, then std gives us the direction to project out of
    # if fixed_dir is resample, then std gives us a [n_vecs, d_model] matrix containing n_vecs orthonormal vectors to project to. We resample within this space 
    def causal_and_save_hook_last_token(self, perturb_type, bsz, std, act_mean, save_to, act, hook):
        act = torch.cat([act, act[:bsz]], dim=0)

        prev_act = act[-bsz:,-1,:].clone()
        if perturb_type == "steer":
            act[-bsz:,-1,:] = act[-bsz:,-1,:] + torch.randn((1,)).to(self.device) * std
        elif perturb_type == "project":
            orig_act = act[-bsz:,-1,:] - act_mean
            standard_dir = std / std.norm()
            act[-bsz:,-1,:] = act[-bsz:,-1,:] - (orig_act * standard_dir).sum() * standard_dir
        elif perturb_type == "resample":
            # std: [n_vecs, d_model], must be orthonormal
            permutation = torch.randperm(bsz)

            # make sure all vecs in batch are resampled
            while (permutation == torch.arange(bsz)).sum() > 0:
                permutation = torch.randperm(bsz)
            permutation = permutation.to(self.device)

            orig_act = act[-bsz:,-1,:] - act_mean
            proj_mtrx = std.permute((1,0)) @ std
            proj_acts = einsum("d_projected d_model, batch d_model -> batch d_projected", proj_mtrx, orig_act)

            # shuffle contributions
            act[-bsz:,-1,:] = act[-bsz:,-1,:] - proj_acts + proj_acts[permutation]
        else:
            norm = torch.randn_like(act[-bsz:,-1,:], dtype=act.dtype).to(self.device)
            mvn_perturb = einsum("d_mvn d_norm, batch d_norm -> batch d_mvn", std.to(act.dtype), norm)
            act[-bsz:,-1,:] = act[-bsz:,-1,:] + mvn_perturb

        save_to.append((prev_act, act[-bsz:,-1,:]))
        return act
    
    # std: [n_layers, d_model]
    def run_causal_perturb(self, batch, std, perturb_type, resample_hook=False):
        n_layers = self.model.cfg.n_layers

        activation_storage = []
        attention_storage = []
        bsz = batch.shape[0]

        target_probs = self.model.run_with_hooks(
                batch,
                fwd_hooks=[
                        *[(partial(resid_points_filter, layer_no), 
                        partial(self.causal_and_save_hook_last_token, 
                                perturb_type, bsz, std[layer_no], self.act_means[layer_no], activation_storage)) 
                        for layer_no in range(n_layers)],

                        # we also need to save attentions for resample
                        *([(partial(attn_out_filter, layer_no), 
                        partial(save_hook_last_token_bsz, bsz, attention_storage))
                        for layer_no in range(n_layers)] if resample_hook else [])
                    ]
        )[:,-1].log_softmax(dim=-1)

        target_probs = target_probs.unflatten(0, (n_layers + 1, bsz)).permute((1,0,2))

        orig_probs = target_probs[:,[0]]
        target_probs = target_probs[:,1:]

        perturb_loss = kl_loss(target_probs, orig_probs.exp()).sum(dim=-1)

        return (target_probs, orig_probs, [a[0] for a in activation_storage], 
                [a[1] for a in activation_storage], perturb_loss, attention_storage if resample_hook else None)

    # perturb_type: ["fixed", "project", False]
    # supported_lens = ["tuned", "linear_oa", "grad", "modal"]
    def get_lens_loss(self, batch, lens_list=["modal", "tuned"], std=0, perturb_type=False, causal_loss=False):

        n_layers = self.model.cfg.n_layers

        output_losses = {}
        a_sims = {}
        causal_losses = {}

        if perturb_type:
            target_probs, orig_probs, orig_acts, activation_storage, output_losses["perturb"], attention_storage = self.run_causal_perturb(batch, std, perturb_type, resample_hook=("resample" in lens_list))
        else:
            activation_storage = []
            attention_storage = []

            target_probs = self.model.run_with_hooks(
                batch,
                fwd_hooks=[
                        *[(partial(resid_points_filter, layer_no), 
                        partial(save_hook_last_token, activation_storage)) 
                        for layer_no in range(n_layers)],

                        # we also need to save attentions for resample
                        *([(partial(attn_out_filter, layer_no), 
                        partial(save_hook_last_token, attention_storage))
                        for layer_no in range(n_layers)] if "resample" in lens_list else [])
                    ]
            )[:,-1].log_softmax(dim=-1).unsqueeze(1)
        
        for k in lens_list:
            if k == "modal" or k == "mean":
                lens_probs = self.apply_modal_lens(k, activation_storage, self.shared_bias or k == "mean")
            elif k == "resample":
                lens_probs = self.apply_modal_lens(k, activation_storage, attention_storage=attention_storage)
            else:
                lens_probs = self.apply_lens(k, activation_storage)

            if perturb_type:
                if k == "modal" or k == "mean":
                    orig_lens_probs = self.apply_modal_lens(k, orig_acts, self.shared_bias or k == "mean")
                elif k == "resample":
                    orig_lens_probs = self.apply_modal_lens(k, orig_acts, attention_storage=attention_storage)
                else:
                    orig_lens_probs = self.apply_lens(k, orig_acts)

                output_losses[k] = kl_loss(lens_probs, orig_lens_probs.exp()).sum(dim=-1)

                # LOGGED PROBS
                a_sims[k] = a_sim((lens_probs - orig_lens_probs).softmax(dim=-1), 
                                (target_probs - orig_probs).softmax(dim=-1), orig_probs.exp())
                if causal_loss:
                    causal_losses[k] = kl_loss(lens_probs, target_probs.exp()).sum(dim=-1)
            else:
                output_losses[k] = kl_loss(lens_probs, target_probs.exp()).sum(dim=-1)

        if causal_loss:
            return output_losses, activation_storage, a_sims, causal_losses
        if perturb_type:
            return output_losses, activation_storage, a_sims
        return output_losses, activation_storage

    def get_vanilla_losses(self, lens_list=["modal", "tuned", "linear_oa", "grad"], no_batches=100, pics_folder=None):
        all_losses = {k: [] for k in lens_list}
        for i in tqdm(range(no_batches)):
            batch = next(self.owt_iter)['tokens']

            with torch.no_grad():
                lens_losses, activation_storage = self.get_lens_loss(batch, lens_list)

                for k in lens_losses:
                    # [batch, n_layers]
                    all_losses[k].append(torch.nan_to_num(lens_losses[k], nan=0, posinf=0, neginf=0))

        # show vanilla losses
        lens_losses_dfs = {}
        lens_losses_dfs = compile_loss_dfs(all_losses, lens_losses_dfs)
        # corr_plot(lens_losses_dfs["modal"], lens_losses_dfs["tuned"], "modal", "tuned")
        # corr_plot(lens_losses_dfs["modal"], lens_losses_dfs["linear_oa"], "modal", "linear_oa")
        corr_plot(lens_losses_dfs["linear_oa"], lens_losses_dfs["tuned"], "linear_oa", "tuned", self.model.cfg.n_layers)
        overall_comp(lens_losses_dfs, title="Lens losses", save=None if pics_folder is None else f"{pics_folder}/original.png")

        means_dfs = {}
        for k in lens_losses_dfs:
            means_dfs[k] = lens_losses_dfs[k].mean()
        return means_dfs

    def get_causal_losses(self, std, perturb_type, batches=100, lens_list=["modal", "linear_oa", "tuned", "grad"], causal_loss=False):
        all_comp_losses = {"perturb": [], **{k: [] for k in lens_list}}
        all_a_sims = {k: [] for k in all_comp_losses}
        all_causal_losses = {k: [] for k in all_comp_losses}
        for i in tqdm(range(batches)):
            batch = next(self.owt_iter)['tokens']

            with torch.no_grad():
                if causal_loss:
                    lens_losses, _, a_sims, causal_loss = self.get_lens_loss(batch, lens_list, std=std, perturb_type=perturb_type, causal_loss=True)
                else:
                    lens_losses, _, a_sims = self.get_lens_loss(batch, lens_list, std=std, perturb_type=perturb_type)

                for k in lens_losses:
                    # [batch, n_layers]
                    all_comp_losses[k].append(torch.nan_to_num(lens_losses[k], nan=0, posinf=0, neginf=0))

                    if k != "perturb":
                        if causal_loss:
                            all_causal_losses[k].append(torch.nan_to_num(causal_loss[k], nan=0, posinf=0, neginf=0))
                        all_a_sims[k].append(torch.nan_to_num(a_sims[k], nan=0, posinf=0, neginf=0))
        
        for k in all_comp_losses:
            all_comp_losses[k] = torch.cat(all_comp_losses[k], dim=0)

            if k != "perturb":
                if causal_loss:
                    all_causal_losses[k] = torch.cat(all_causal_losses[k], dim=0)
                all_a_sims[k] = torch.cat(all_a_sims[k], dim=0)
        
        # comp_loss: lens KL vs original prediction of lens. causal_loss: lens KL vs NEW prediction of model.
        if causal_loss:
            return all_comp_losses, all_a_sims, all_causal_losses
        else:
            return all_comp_losses, all_a_sims

    def get_causal_perturb_losses(self, lens_list=["modal", "linear_oa", "tuned", "grad"], kl_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5, 1], save=None, pics_folder=None, plotting=True):
        causal_losses = {}
        for t in kl_thresholds:
            causal_magnitudes = self.retrieve_causal_mag(t)[:, None, None] * self.a_mtrx
            causal_losses[t] = self.get_causal_losses(causal_magnitudes, "perturb", lens_list=lens_list, causal_loss=True)

        if save:
            torch.save(causal_losses, save)

        if not plotting:
            return

        for t in causal_losses: 
            causal_lens_losses_dfs = {}
            causal_lens_losses_dfs = compile_loss_dfs(causal_losses[t][2], causal_lens_losses_dfs, suffix="_causal")
            # causal_lens_losses_dfs = {**causal_lens_losses_dfs, **lens_losses_dfs}
            overall_comp(causal_lens_losses_dfs, title=f"Lens losses causal {t}", save=None if pics_folder is None else f"{pics_folder}/causal_{t}.png")

        for t in causal_losses: 
            causal_lens_losses_dfs = {}
            if 'modal' in causal_losses[t][1]:
                del causal_losses[t][1]['modal']
            if 'perturb' in causal_losses[t][1]:
                del causal_losses[t][1]['perturb']
            
            causal_lens_losses_dfs = compile_loss_dfs(causal_losses[t][1], causal_lens_losses_dfs, suffix="_causal")
            overall_comp(causal_lens_losses_dfs, title=f"Lens losses A-similarity {t}", save=None if pics_folder is None else f"{pics_folder}/causal_sim_{t}.png")

    def get_perturb_magnitudes(self, folder):
        # For perturb loss, get std threshold for reaching a certain level of KL loss
        if os.path.exists(f"{folder}/perturb_losses.pkl"):
            with open(f"{folder}/perturb_losses.pkl", "rb") as f:
                perturb_losses = pickle.load(f)
        else:
            perturb_losses = []

            with torch.no_grad():
                for std in self.std_intvs:
                    perturb_losses_by_std = []
                    for i in tqdm(range(200)):
                        batch = next(self.owt_iter)['tokens']

                        # pass in MVN transformation
                        _, _, _, _, perturb_loss, _ = self.run_causal_perturb(batch, std * self.std_mult[..., None, None] * self.a_mtrx, perturb_type="perturb")
                        
                        perturb_losses_by_std.append(perturb_loss.mean(dim=0))
                    perturb_losses_by_std = torch.stack(perturb_losses_by_std, dim=0).mean(dim=0)

                    print(perturb_losses_by_std)
                    perturb_losses.append(perturb_losses_by_std)

            with open(f"{folder}/perturb_losses.pkl", "wb") as f:
                pickle.dump(perturb_losses, f)

        perturb_losses = torch.stack(perturb_losses, dim=-1)
        return perturb_losses

    def retrieve_causal_mag(self, t):
        above_t = (torch.arange(self.perturb_losses.shape[-1], device=self.device).repeat(self.perturb_losses.shape[0],1) * (self.perturb_losses < t)).argmax(dim=-1)
        causal_magnitudes = self.perturb_mag[torch.arange(self.perturb_mag.shape[0]).to(self.device), above_t]
        return causal_magnitudes


# %%
