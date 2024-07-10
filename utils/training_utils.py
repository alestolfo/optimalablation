# %%

import torch
from transformer_lens import HookedTransformer
from itertools import cycle
import torch.optim
from fancy_einsum import einsum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math
import argparse
from utils.data import retrieve_owt_data


# %%
    
default_args = {
    "name": None,
    "lamb": 1e-3,
    "dataset": "ioi",
    "subfolder": None,
    "priorscale": None,
    "priorlamb": None,
    "desc": None
}

def load_args(run_type, default_lamb=None, defaults={}):
    my_args = {**default_args, **defaults, "lamb": default_lamb}
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', '--lamb',
                            help='regularization constant')
        parser.add_argument('-d', '--dataset',
                            help='ioi or gt')
        parser.add_argument('-s', '--subfolder',
                            help='where to load/save stuff')
        parser.add_argument('-c', '--priorscale',
                            help='prior strength')
        parser.add_argument('-p', '--priorlamb',
                            help='which vertex lambda')
        parser.add_argument('-n', '--name',
                            help='run name, e.g. edges or vertex prior')
        parser.add_argument('-t', '--tau',
                            help='threshold to use for post training')
        parser.add_argument('-e', '--desc',
                            help='ablation type')
        parser.add_argument('-w', '--window',
                            help='dynamic-window',
                            default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--minwindow',
                            help='minwindow')
        parser.add_argument('--maxwindow',
                            help='maxwindow')

        args = parser.parse_args()
        args = vars(args)

        for k in args:
            my_args[k] = args[k]
            if k in {"lamb", "priorscale", "priorlamb", 
                     "tau", "minwindow", "maxwindow"} and my_args[k] is not None:
                # exception for manual circuit
                if my_args[k] == "manual":
                    continue
                my_args[k] = float(my_args[k])
    except Exception as e:
        print(e)
        print("Resetting to default parameters")
        pass
    except:
        print("Resetting to default parameters")
        pass

    print(my_args["lamb"])
    parent = "results"

    run_folder = (my_args["dataset"] if my_args["name"] is None 
                  else f"{my_args['dataset']}/{my_args['name']}" if my_args["desc"] is None
                  else f"{my_args['dataset']}/{my_args['desc']}/{my_args['name']}")
    if my_args["subfolder"] is not None:
        folder=f"{parent}/{run_type}/{run_folder}/{my_args['subfolder']}"
    elif my_args["priorlamb"] is not None:
        folder=f"{parent}/{run_type}/{run_folder}/{my_args['lamb']}-{my_args['priorlamb']}-{my_args['priorscale']}"
    elif my_args["lamb"] is None:
        folder=f"{parent}/{run_type}/{run_folder}"
    else:
        folder=f"{parent}/{run_type}/{run_folder}/{my_args['lamb']}"

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    my_args["folder"] = folder
    return my_args

def load_model_data(model_name, batch_size=8, ctx_length=25, repeats=True, ds_name=False, device="cuda:0"):
    # device="cpu"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    print("Loading OWT...")
    try:
        if ds_name:
            owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer, ds_name=ds_name)
        else:
            owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer)
        if repeats:
            owt_iter = cycle(owt_loader)
        else:
            owt_iter = owt_loader
    except:
        owt_iter = None
    return device, model, tokenizer, owt_iter

# %%

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"
attn_out_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_attn_out"

def gen_resample_perm(n):
    permutation = torch.randperm(n)
    # make sure all prompts are resampled
    while (permutation == torch.arange(n)).sum() > 0:
        permutation = torch.randperm(n)
    
    return permutation

def save_hook_last_token(save_to, act, hook):
    save_to.append(act[:,-1,:])

def save_hook_last_token_bsz(bsz, save_to, act, hook):
    save_to.append(act[:bsz,-1,:])

def save_hook_all_tokens(save_to, act, hook):
    save_to.append(act)

def ablation_hook_last_token(batch_feature_idx, repl, act, hook):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    # act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)[batch_feature_idx[:,0],batch_feature_idx[:,1]]
    act = act[batch_feature_idx]
    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    # act = torch.cat([act,torch.zeros(1,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act
    # return act.repeat(features_per_batch,1,1)
    # pass

def ablation_all_hook_last_token(repl, act, hook):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    # act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)[batch_feature_idx[:,0],batch_feature_idx[:,1]]
    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    # act = torch.cat([act,torch.zeros(1,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act
    # return act.repeat(features_per_batch,1,1)
    # pass

def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def ablation_hook_attention_all_tokens(constants, bsz, activation_storage, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(constants.shape[0]):
        # if attentions.shape[0] > 400:
        # print(start)
        attentions[-start:-start+bsz,:,i] = constants[i].clone()
        start -= bsz
    
    # print(attentions.shape)
    # if attentions.shape[0] > 400:
    #     sns.histplot(attentions[:bsz][attentions[:bsz].abs() > 20].detach().flatten().cpu())
    #     print((attentions[:bsz].abs() > 500).nonzero())
    #     print(attentions[:bsz][(attentions[:bsz].abs() > 500)])
        
    # ignore first token because it is crazy
    with torch.no_grad():
        activation_storage.append(attentions[:bsz,1:].mean(dim=[0,1]))
    return attentions

# attentions: (batch_size + batch_size * n_samples) x seq_len x n_heads x d_model
# constants: n_heads x d_model
# prune mask: (batch_size * n_samples) x n_heads, 0 = prune, 1 = keep
def pruning_hook_attention_all_tokens(constants, prune_mask, bsz, attentions, hook):
    # N by 2. First column = batch item, second column = head idx
    prune_mask = prune_mask.unsqueeze(1).unsqueeze(-1)
    attentions[bsz:] = (1-prune_mask) * constants + prune_mask * attentions[bsz:].clone()

    # prune_idx = prune_mask.clone()
    # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
    return attentions

def tuned_lens_hook(activation_storage, tuned_lens_weights, tuned_lens_bias, act, hook):
    activation_storage.append(einsum("result activation, batch activation -> batch result", tuned_lens_weights, act[:,-1]) + tuned_lens_bias)
    return act

# rec = number of items to record
# prev_means: rec x 1
# prev_vars: rec x 1
# batch_results: rec x n_samples
# no batches: number of batches represented in prev_means and prev_variances
# 
def update_means_variances(prev_means, prev_vars, batch_results, no_batches):
    # computing variance iteratively using a trick
    old_samples = batch_results.shape[-1] * no_batches
    new_samples = batch_results.shape[-1] * (no_batches + 1)

    means = (no_batches * prev_means + batch_results.mean(dim=-1, keepdim=True)) / (no_batches + 1)

    prev_vars = prev_vars * (old_samples - 1)

    vars = prev_vars + (batch_results - prev_means).square().sum(dim=-1, keepdim=True) - new_samples * (means - prev_means).square()
    
    if new_samples > 1:
        vars = vars / (new_samples - 1)
    return means, vars

# rec = number of items to record
# prev_means: rec x 1
# prev_vars: rec x 1
# batch_results: rec x n_samples (rec x 1 if just one batch). Assumed to be mean, not sum.
# n_batches_by_head: rec x 1 (previous no. batches)
# n_samples_by_head: rec x 1 (previous no. samples)
# batch_samples_by_head: rec x n_samples (rec x 1 if just one batch)
def update_means_variances_mixed(prev_means, prev_vars, batch_results, n_batches_by_head, n_samples_by_head, batch_samples_by_head):
    # computing variance iteratively using a trick
    new_batches_by_head = n_batches_by_head + (batch_samples_by_head > 0).sum(dim=-1, keepdim=True)
    new_samples_by_head = n_samples_by_head + batch_samples_by_head.sum(dim=-1, keepdim=True)

    means = prev_means * n_samples_by_head
    means = means + (batch_samples_by_head * batch_results).sum(dim=-1, keepdim=True)
    means = torch.where(
        new_samples_by_head > 0,
        means / new_samples_by_head,
        means
    )

    prev_vars = prev_vars * (n_batches_by_head - 1)
    vars = prev_vars + (batch_samples_by_head * (batch_results - prev_means).square()).sum(dim=-1, keepdim=True) - new_samples_by_head * (means - prev_means).square()


    # print(batch_samples_by_head * (batch_results - prev_means).square())
    # print(new_samples_by_head * (means - prev_means).square())
    # print(batch_samples_by_head)
    # print(means)
    # print(prev_means)
    # print(new_batches_by_head)
    # print(vars)

    # print((batch_samples_by_head * (batch_results - prev_means).square()).sum(dim=-1, keepdim=True) - new_samples_by_head * (means - prev_means).square())

    vars = torch.where(
        new_batches_by_head > 1,
        vars / (new_batches_by_head - 1),
        vars
    )
    
    return means, vars, new_batches_by_head, new_samples_by_head

# %%

# only accepts a single batch at a time
# n_batches_by_head should be initialized to 0

# rec = number of items to record
# prev_means: rec x 1
# prev_vars: rec x 1
# batch_results: rec x 1. Assumed to be mean, not sum.
# n_batches_by_head: rec x 1 (previous no. batches)
# n_samples_by_head: rec x 1 (previous no. samples)
# batch_samples_by_head: rec x 1
def update_means_variances_exponential(prev_means, prev_vars, batch_results, n_batches_by_head, n_samples_by_head, batch_samples_by_head, total_num_batches, alpha=0.95):

    # Avg samples per batch, including instances of zero samples
    new_samples_by_head = alpha * n_samples_by_head + (1-alpha) * batch_samples_by_head

    # divide by probability mass represented by 0s
    avg_samples_by_head = new_samples_by_head / (1 - math.exp(math.log(alpha) * (total_num_batches + 1)))

    new_w = torch.where(
        n_samples_by_head > 0,
        
        # then cap how much you update on any one batch
        (math.log(alpha) * (batch_samples_by_head / avg_samples_by_head).clip(max=2)).exp(),
        alpha
    )

    # total mass of exponential-weighted sequence. If new_samples_by_head=0, then it should be 1.
    new_batches_by_head = 1 - (1 - n_batches_by_head) * new_w

    # undo division
    means = prev_means * n_batches_by_head
    means = new_w * means + (1-new_w) * batch_results
    # print(new_w[:3])
    # print(((1-new_w) * batch_results)[:3])


    # undo division
    prev_vars = prev_vars * n_batches_by_head
    vars = torch.where(
        n_batches_by_head == 0,
        0,
        new_w * prev_vars + (1-new_w) * batch_samples_by_head * (batch_results - prev_means).square()
    )

    # prev_vars = prev_vars * n_batches_by_head
    # vars = new_w * prev_vars + (1-new_w) * (batch_results - prev_means).square() - (means - prev_means).square()
    # print(vars)
    # print(new_batches_by_head.mean())
    # vars = torch.where(
    #     vars < 0,
    #     0,
    #     vars
    # )
    # print("Previous mean", prev_means[144])
    # print(vars[144])
    # print(means[144])
    # print(((1-new_w) * batch_samples_by_head * (batch_results - prev_means).square())[144])

    means = torch.where(
        new_batches_by_head > 0,
        means / new_batches_by_head,
        means
    )
    vars = torch.where(
        new_batches_by_head > 0,
        vars / new_batches_by_head,
        vars
    )
    return means, vars, new_batches_by_head, new_samples_by_head


# %%

def plot_no_outliers(plot_fn, alpha, x, y, ax=None, xy_line=False, args={}):
    if isinstance(x, torch.Tensor):
        x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0).detach().cpu().flatten()
    
    if isinstance(y, torch.Tensor):
        y = torch.nan_to_num(y, nan=0, posinf=0, neginf=0).detach().cpu().flatten()
    
    if alpha > 0:
        x_sat = (x > x.quantile(alpha)) * (x < x.quantile(1-alpha))
        y_sat = (y > y.quantile(alpha)) * (y < y.quantile(1-alpha))
        x = x[x_sat * y_sat]
        y = y[x_sat * y_sat]

    plot_args = {"x": x, "y": y, "ax": ax}
    if "s" in args:
        plot_args["s"] = args["s"]
    cur_ax = plot_fn(**plot_args)

    if xy_line:
        min_val = max(cur_ax.get_xlim()[0],cur_ax.get_ylim()[0])
        max_val = min(cur_ax.get_xlim()[1],cur_ax.get_ylim()[1])
        cur_ax.plot([min_val, max_val],[min_val, max_val], color="red", linestyle="-")
    
    corr = 0
    if "x" in args:
        cur_ax.set_xlabel(args["x"])
    if "y" in args:
        cur_ax.set_ylabel(args["y"])
    if "title" in args:
        cur_ax.set_title(args["title"])
    if "corr" in args:
        corr = round(np.corrcoef(x,y)[0,1],3)
        ax.text(.05, .8, f"r={corr}", transform=cur_ax.transAxes)
    if "f" in args:
        plt.savefig(args["f"])
        plt.close()
    return corr


# %%
# Unit test for update means variances mixed
# true_means = (torch.randn((100,1)) * 50).to(device)
# true_vars = (torch.randn((100,1)).abs() * 50).to(device)

# est_means = torch.zeros_like(true_means).to(device)
# est_vars = torch.zeros_like(true_means).to(device)
# n_batches_by_head = torch.zeros_like(true_means).to(device)
# n_samples_by_head = torch.zeros_like(true_means).to(device)

# for b in range(100):
#     mean_samples = []
#     sample_counts = []
#     for s in range(5):
#         n_samples = (torch.randint(0,10,(100,1)) - 5).relu().to(device)
#         idx_arr = torch.arange(10).unsqueeze(0).repeat(100,1).to(device)
#         idx_mask = (idx_arr < n_samples) * 1

#         batch_samples = true_vars.sqrt() * torch.randn((100,10)).to(device) + true_means
#         batch_means = torch.where(
#             n_samples < 1, 
#             0,
#             (batch_samples * idx_mask).sum(dim=-1, keepdim=True) / n_samples
#         )
#         mean_samples.append(batch_means)
#         sample_counts.append(n_samples)
#     mean_samples = torch.cat(mean_samples, dim=1) 
#     sample_counts = torch.cat(sample_counts, dim=1)

#     est_means, est_vars, n_batches_by_head, n_samples_by_head = update_means_variances_mixed(est_means, est_vars, mean_samples, n_batches_by_head, n_samples_by_head, sample_counts)

#     if b % -10 == -1:
#         sns.scatterplot(x=est_vars.flatten().cpu(), y=true_vars.flatten().cpu())
#         sns.lineplot(x=[0,200], y=[0,200])
#         plt.show()

#         sns.scatterplot(x=est_means.flatten().cpu(), y=true_means.flatten().cpu())
#         sns.lineplot(x=[-200,200], y=[-200,200])
#         plt.show()

def clip_grads(params, bound):
    grad_norms = []
    for param in params:
        grad_norm = torch.nn.utils.clip_grad_norm_(param, max_norm=float('inf'))
        grad_norms.append(grad_norm.item())
        torch.nn.utils.clip_grad_norm_(param, max_norm=bound)
    return grad_norms

class LinePlot:
    def __init__(self, stat_list, pref_start=100):
        self.stat_list = stat_list
        self.stat_book = {x: [] for x in stat_list}
        self.t = 0
        self.last_tick = 0
        self.early_term_count = 0
        self.pref_start = pref_start
    
    def add_entry(self, entry):
        for k in self.stat_book:
            if k in entry:
                self.stat_book[k].append(entry[k])
            # default behavior is flat line
            elif self.t == 0:
                self.stat_book[k].append(0)
            else:
                self.stat_book[k].append(self.stat_book[k][-1])
        self.t += 1
    
    def mv_avg(self, series, mv=50):
        yvals = self.stat_book[series]
        return [np.mean(yvals[i:min(len(yvals),i+mv)]) for i in range(len(yvals))]
    
    def stat_sig_growth(self, series, avg_intv=10, comp_intv=200, start_t=0):
        if self.t - start_t <= comp_intv + avg_intv + 1:
            return False
        historical_avg = [np.mean(self.stat_book[series][-i-avg_intv-1:-i-1]) for i in range(comp_intv // 2, comp_intv, (avg_intv // 3))]
        rolling_avg = np.mean(self.stat_book[series][-avg_intv:])

        # decline, growth
        return 1 - rolling_avg / np.quantile(historical_avg, .1), rolling_avg / np.quantile(historical_avg, .9) - 1
    
    def compare_plot(self, series, mv, compare_log, title="", start=100):
        max_t = min(self.t, compare_log.t)
        sns.lineplot(self.mv_avg(series, mv=mv)[start:], label="control (first arg)")
        sns.lineplot(compare_log.mv_avg(series, mv=mv)[start:], label="new")
        plt.legend()
        plt.title(title)
        plt.minorticks_on()
        plt.grid(visible=True, which='major', color='k', linewidth=1)
        plt.grid(visible=True, which='minor', color='k', linewidth=0.5)
        plt.show()

    def plot(self, series=None, subplots=None, step=1, start=None, end=0, agg='mean', twinx=True, mv=False, save=None, gridlines=False):
        if start is None:
            start = self.pref_start
        if series is None:
            series = self.stat_list
        if end <= start:
            end = self.t
            if end <= start:
                start = 0
        t = [i for i in range(start, end, step)]
        ax = None
        (h,l) = ([],[])
        colors = ["green", "blue", "red", "orange"]
        if subplots is not None:
            rows = (len(series)-1) // subplots + 1
            f, axes = plt.subplots(rows, subplots, figsize=(subplots * 5, rows * 5))
            
        for i,s in enumerate(series):
            if agg == 'mean':
                yvals = [np.mean(self.stat_book[s][i:i+step]) for i in range(start, end, step)]
            else:
                yvals = [self.stat_book[s][i] for i in range(start, end, step)]
            if twinx is True:
                params = {"x": t, "y": yvals, "label": s}
                if len(series) <= 4:
                    params["color"] = colors[i]
                if ax is None:
                    ax = sns.lineplot(**params)
                    h, l = ax.get_legend_handles_labels()
                    ax.get_legend().remove()
                    cur_ax = ax
                else:
                    ax2 = sns.lineplot(**params, ax=ax.twinx())
                    ax2.get_legend().remove()
                    h2, l2 = ax2.get_legend_handles_labels()
                    h += h2
                    l += l2
                    cur_ax = ax
            else:
                ax = sns.lineplot(x=t, y=yvals, label=s, ax=None if subplots is None else axes[i // subplots, i % subplots])
                cur_ax = ax
            if mv:
                mv_series = [np.mean(yvals[i:min(len(yvals),i+mv)]) for i in range(len(yvals))]
                sns.lineplot(x=t, y=mv_series, label=f"{s}_mv_{mv}", ax=cur_ax)
        if gridlines:
            plt.minorticks_on()
            plt.grid(visible=True, which='major', color='k', linewidth=1)
            plt.grid(visible=True, which='minor', color='k', linewidth=0.5)

        if h is None:
            plt.legend()
        else:
            plt.legend(h, l)
        plt.tight_layout()

        if save:
            plt.savefig(save)
        plt.show()
        plt.close()

    def export():
        pass

