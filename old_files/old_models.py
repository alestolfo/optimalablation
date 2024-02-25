# %%
import torch
from transformer_lens import HookedTransformer
from data import retrieve_owt_data
from itertools import cycle
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
import math
from functools import partial
import torch.optim
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(model_name, device=device)

# %%

tokenizer = model.tokenizer
batch_size = 8
max_context_length = 35
owt_loader = retrieve_owt_data(batch_size, max_context_length, tokenizer)
owt_iter = cycle(owt_loader)

# %%

# inverse probe setting
layer_no = 2
num_features = 2000
activation_dim = 768
features_per_batch = 75

# learning hyperparameters
convergence_tol = .01
similiarity_tol = .05
lr_act = .1

# %%

intervene_filter = lambda name: name == f"blocks.{layer_no}.hook_resid_post"
relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

def save_hook_last_token(save_to, act, hook):
    save_to.append(act[:,-1,:])

def ablation_hook_last_token(batch_feature_idx, repl, act, hook):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)[batch_feature_idx[:,0],batch_feature_idx[:,1]]

    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    # act = torch.cat([act,torch.zeros(1,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act
    # return act.repeat(features_per_batch,1,1)
    # pass

# last_step_sz nullable
def poly_scheduler(t, similarity):
    return .5 * t ** .5 * relu(similarity - convergence_tol).square().sum()

# %%
def plot_curves(converged, penalties, feat_similarities):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    line1 = sns.lineplot(x=range(len(converged)), y=converged, ax=ax1, label="convergence", color="blue")
    line2 = sns.lineplot(x=range(len(converged)), y=penalties, ax=ax1, label="feature penalty", color="red")
    line3 = sns.lineplot(x=range(len(converged)), y=feat_similarities, ax=ax2, label="avg feature similarity", color="orange")

    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    ax1.legend(handles, labels, loc='upper right')
    ax2.get_legend().set_visible(False)

    plt.show()

# start by assuming the model is perfect
def train_activations(model, feature_directions, reg_scheduler, verbose=False):
    with torch.no_grad():
        model.eval()
        
        # save the original activations
        cur_activations = []

        # -1 gives last token
        target_probs = model.run_with_hooks(batch, fwd_hooks=[(intervene_filter, partial(save_hook_last_token, cur_activations))])[:,-1].softmax(dim=-1)

        cur_activations = cur_activations[0]

    for param in model.parameters():
        param.requires_grad = False
    
    optim_activations = []
    agg_losses = []

    # initial_similarities = einsum(
    #     "batch d_model, feature d_model -> batch feature", 
    #     cur_activations,
    #     feature_directions
    # )

    num_batches = math.ceil(num_features / features_per_batch)
    for j in tqdm(range(num_batches)):
        start_batch = j * features_per_batch
        end_batch = (j+1) * features_per_batch

        opt_act = torch.nn.Parameter(cur_activations.unsqueeze(1).repeat(1, features_per_batch, 1))
        rvlt_directions = feature_directions[start_batch:end_batch]
        optimizer = torch.optim.SGD([opt_act], lr=lr_act, weight_decay=0)

        last_step_sz = None
        t = 0

        converged = []
        penalties = []
        av_feature_similarities = []

        # flag whether specific activations have converged
        continue_training = torch.ones((batch_size, features_per_batch)).to(device)
        final_losses = torch.zeros((batch_size, features_per_batch)).to(device)
        initial_similarities = None

        while continue_training.sum() > 0:
            optimizer.zero_grad()

            # N x 2, N is the number of (batch_dim, feature_dim) tuples that need to continue training
            continue_idx = continue_training.nonzero()
            converged.append(continue_idx.shape[0])
            # opt_act: batch features d_model
            prev_act = opt_act[continue_idx[:,0],continue_idx[:,1]].data.detach()

            # (batch * feature) x seq_len (-1 gives last token) x vocab_size
            cur_probs = model.run_with_hooks(
                batch, 
                fwd_hooks=[(intervene_filter, 
                            partial(ablation_hook_last_token,
                                    continue_idx, 
                                    opt_act[continue_idx[:,0],continue_idx[:,1]])
                            )]
            )[:,-1].softmax(dim=-1)

            kl_losses = kl_loss(cur_probs.log(), target_probs[continue_idx[:,0]]).sum(dim=-1)

            # sns.histplot(kl.detach().cpu().numpy())
            # plt.show()
            # loss = kl.sum()

            # sns.histplot(torch.abs(cur_probs - target_probs[continue_idx[:,0]]).sum(dim=1).detach().cpu().numpy())
            # return

            feature_similarities = einsum(
                "batch_feature d_model, batch_feature d_model -> batch_feature", 
                opt_act[continue_idx[:,0],continue_idx[:,1]], 
                rvlt_directions[continue_idx[:,1]]
            )
            if initial_similarities is None:
                initial_similarities = feature_similarities.detach()
            penalty = reg_scheduler(t, feature_similarities)
            penalties.append(penalty.item())
            av_feature_similarities.append(relu(feature_similarities).mean().item())

            loss = kl_losses.sum() + penalty
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                last_step_sz = (prev_act - opt_act[continue_idx[:,0],continue_idx[:,1]]).norm(dim=-1)

                # print(opt_act.grad.shape)

                # sns.scatterplot(x=last_step_sz.flatten().cpu().numpy(),y=feature_similarities.flatten().cpu().numpy())
                # plt.show()

                # return opt_act.grad, opt_act, prev_act
                
                # stop when it has converged AND sufficiently non-feature-related
                # N x 2, N is the number of (batch_dim, feature_dim) tuples to stop training
                stop_train = (torch.min(last_step_sz < convergence_tol, feature_similarities < similiarity_tol)).nonzero()[:,0].to(device)
                # print((feature_similarities > similiarity_tol).nonzero().shape[0])
                # print((last_step_sz > convergence_tol).nonzero().shape[0])
                # print(last_step_sz.min())
                # sns.histplot(last_step_sz.flatten().cpu().numpy())
                # plt.show()

                # sns.histplot(feature_similarities.flatten().cpu().numpy())
                # plt.show()

                # print(continue_training[continue_idx[:,0], continue_idx[:,1]].shape)
                # print(stop_train)
                # return
                continue_training[continue_idx[stop_train,0], continue_idx[stop_train,1]] = 0
                final_losses[continue_idx[stop_train,0], continue_idx[stop_train,1]] = kl_losses.detach()[stop_train]
            t += 1 
            if verbose and t % 100 == 0:
                plot_curves(converged,penalties,av_feature_similarities)
                sns.scatterplot(x=feature_similarities.detach().flatten().cpu().numpy(), y=last_step_sz.detach().flatten().cpu().numpy())
                plt.show()
        if verbose:
            plot_curves(converged,penalties,av_feature_similarities)

            sns.scatterplot(x=initial_similarities.cpu().numpy(), y=final_losses.flatten().cpu().numpy())
            plt.show()

        optim_activations.append(opt_act)
        agg_losses.append(final_losses)
    return torch.cat(optim_activations, dim=1), torch.cat(agg_losses, dim=1)

# one activation vec for each feature
# potentially very large

# %%

# need to do a smarter thing with a smaller model maybe?
repl = torch.zeros(batch_size, activation_dim)

feature_directions = torch.normal(0,1,(num_features, activation_dim)).to(device)
feature_directions = feature_directions / feature_directions.norm(dim=-1, keepdim=True)
# %%
for i in range(1):
    batch = next(owt_iter)['tokens']
    
    states = train_activations(model, feature_directions, poly_scheduler)

    # kl_loss(abl_logits, target_logits)

# %%
