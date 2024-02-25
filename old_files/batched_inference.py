# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, save_hook_last_token, ablation_hook_last_token

# %%


# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 8
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

# inverse probe setting
layer_no = 6
num_features = 2000
activation_dim = 768
features_per_batch = 50 * batch_size

# learning hyperparameters
convergence_tol = .005
similarity_tol = .05
lr_act = .05
top_k = 20
relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

intervene_filter = lambda name: name == f"blocks.{layer_no}.hook_resid_post"

# last_step_sz nullable
def poly_scheduler(t, similarity):
    return .5 * (t + 10) * relu(similarity - convergence_tol).sum()

# %%
def plot_curves(converged, penalties, feat_similarities):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    line1 = sns.lineplot(x=range(len(converged)), y=converged, ax=ax1, label="convergence", color="blue")
    line2 = sns.lineplot(x=range(len(converged)), y=penalties, ax=ax2, label="feature penalty", color="red")
    line3 = sns.lineplot(x=range(len(converged)), y=feat_similarities, ax=ax3, label="avg feature similarity", color="orange")

    handles, labels = [], []
    for ax in [ax1, ax2, ax3]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    ax1.legend(handles, labels, loc='upper right')
    ax2.get_legend().set_visible(False)
    ax3.get_legend().set_visible(False)

# %%

def linear_proj(starting_activations, rvlt_directions):
    return starting_activations - (starting_activations * rvlt_directions).sum(dim=-1, keepdim=True) * rvlt_directions / rvlt_directions.norm(dim=-1, keepdim=True)

# start by assuming the model is perfect
def train_activations(model, feature_directions, reg_scheduler, verbose=False):
    def f_h(batch, batch_pairs, linear_projections, rvlt_directions, kth_top_loss, final_losses, convergence_times, reg_scheduler, start_t=0, soft_limit=10, hard_limit=10, initial_losses=None):
        opt_act = torch.nn.Parameter(linear_projections)
        optimizer = torch.optim.SGD([opt_act], lr=lr_act, weight_decay=0)

        last_step_sz = None
        t = start_t

        converged = []
        penalties = []
        av_feature_similarities = []

        # flag whether specific activations have converged
        # continue_training = torch.ones((batch_size, features_per_batch)).to(device)
        # final_losses = torch.zeros((batch_size, features_per_batch)).to(device)
        continue_training = torch.ones((linear_projections.shape[0],)).to(device)

        # print(time.time_ns() - start_time)
        # start_time = time.time_ns()
        remainder = continue_training.sum()
        while remainder > 0 and (t < soft_limit or remainder >= 10) and t < hard_limit:
            optimizer.zero_grad()

            # N x 2, N is the number of (batch_dim, feature_dim) tuples that need to continue training
            continue_idx = continue_training.nonzero()[:,0]
            converged.append(continue_idx.shape[0])
            # opt_act: batch features d_model
            # prev_act = opt_act[continue_idx[:,0],continue_idx[:,1]].data.detach()
            prev_act = opt_act[continue_idx].data.detach()

            # (batch * feature) x seq_len (-1 gives last token) x vocab_size
            # cur_probs = model.run_with_hooks(
            #     batch, 
            #     fwd_hooks=[(intervene_filter, 
            #                 partial(ablation_hook_last_token,
            #                         continue_idx, 
            #                         opt_act[continue_idx[:,0],continue_idx[:,1]])
            #                 )]
            # )[:,-1].softmax(dim=-1)

            cur_probs = model.run_with_hooks(
                batch, 
                fwd_hooks=[(intervene_filter, 
                            partial(ablation_hook_last_token,
                                    batch_pairs[0][continue_idx],
                                    opt_act[continue_idx])
                            )]
            )[:,-1].softmax(dim=-1)
            
            # print(time.time_ns() - start_time)
            # start_time = time.time_ns()

            # kl_losses = kl_loss(cur_probs.log(), target_probs[continue_idx[:,0]]).sum(dim=-1)
            kl_losses = kl_loss(cur_probs.log(), target_probs[batch_pairs[0][continue_idx]]).sum(dim=-1)

            if initial_losses is not None and t == 0:
                initial_losses[batch_pairs[0], batch_pairs[1]] = kl_losses.detach()

            # sns.histplot(torch.abs(cur_probs - target_probs[continue_idx[:,0]]).sum(dim=1).detach().cpu().numpy())
            # return
            # print(batch_pairs[1].shape)

            feature_similarities = einsum(
                "batch_feature d_model, batch_feature d_model -> batch_feature", 
                opt_act[continue_idx],
                rvlt_directions[continue_idx]
            )
            
            # print(time.time_ns() - start_time)
            # start_time = time.time_ns()

            penalty = reg_scheduler(t, feature_similarities)
            penalties.append(penalty.item())
            av_feature_similarities.append(relu(feature_similarities).mean().item())

            loss = kl_losses.sum() + penalty
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                last_step_sz = (prev_act - opt_act[continue_idx]).norm(dim=-1)

                # print(opt_act.grad.shape)

                # sns.scatterplot(x=last_step_sz.flatten().cpu().numpy(),y=feature_similarities.flatten().cpu().numpy())
                # plt.show()

                # return opt_act.grad, opt_act, prev_act
                
                # stop when it has converged AND sufficiently non-feature-related
                # N x 2, N is the number of (batch_dim, feature_dim) tuples to stop training
                # stop_train = (torch.min(last_step_sz < convergence_tol, feature_similarities < similarity_tol)).nonzero()[:,0].to(device)
                # print(top_k_losses[batch_pairs[0][continue_idx]].shape)
                stop_train = (torch.min(feature_similarities < similarity_tol, torch.max(last_step_sz < convergence_tol, kl_losses < kth_top_loss[batch_pairs[0][continue_idx]]))).nonzero().to(device)
                # print((feature_similarities > similarity_tol).nonzero().shape[0])
                # print((last_step_sz > convergence_tol).nonzero().shape[0])
                # print(last_step_sz.min())
                # sns.histplot(last_step_sz.flatten().cpu().numpy())
                # plt.show()

                # sns.histplot(feature_similarities.flatten().cpu().numpy())
                # plt.show()

                # print(continue_training[continue_idx[:,0], continue_idx[:,1]].shape)
                # print(stop_train)
                # return
                continue_training[continue_idx[stop_train]] = 0
                # final_losses[continue_idx[stop_train,0], continue_idx[stop_train,1]] = kl_losses.detach()[stop_train]
                final_losses[batch_pairs[0][continue_idx[stop_train]],batch_pairs[1][continue_idx[stop_train]]] = kl_losses.detach()[stop_train]
                convergence_times[batch_pairs[0][continue_idx[stop_train]],batch_pairs[1][continue_idx[stop_train]]] = t
            # print(time.time_ns() - start_time)
            # start_time = time.time_ns()
            remainder = continue_training.sum()
            t += 1 
        if t > 10:
            print("convergence:", t)
        return opt_act, batch_pairs[:,continue_training.nonzero()[:,0]], converged, penalties, av_feature_similarities

    with torch.no_grad():
        model.eval()
        
        # save the original activations
        cur_activations = []

        # -1 gives last token
        target_probs = model.run_with_hooks(batch, fwd_hooks=[(intervene_filter, partial(save_hook_last_token, cur_activations))])[:,-1].softmax(dim=-1)

        cur_activations = cur_activations[0]

    for param in model.parameters():
        param.requires_grad = False
    
    # batch-feature x 2
    initial_similarities = einsum(
        "batch d_model, feature d_model -> batch feature", 
        cur_activations,
        feature_directions
    )
    optim_activations = cur_activations.unsqueeze(1).repeat(1, num_features, 1)
    final_losses = torch.zeros((batch_size,num_features)).to(device)
    initial_losses = torch.zeros((batch_size,num_features)).to(device)
    convergence_times = torch.zeros((batch_size,num_features)).to(device)
    kth_top_loss = torch.zeros((batch_size,)).to(device)

    # start_time = time.time_ns()

    agg_converged = []
    agg_penalties = []
    agg_feature_sim = []
    longest_convergence = 0
    second_pass = []

    activations_to_train = (initial_similarities > similarity_tol).nonzero().permute(1,0)
    num_batches = math.ceil((activations_to_train.shape[1]) / features_per_batch)

    for j in tqdm(range(num_batches)):
        start_batch = j * features_per_batch
        end_batch = (j+1) * features_per_batch

        # opt_act = torch.nn.Parameter(cur_activations.unsqueeze(1).repeat(1, features_per_batch, 1))

        batch_pairs = activations_to_train[:,start_batch:end_batch]
        rvlt_directions = feature_directions[batch_pairs[1]]
        starting_activations = cur_activations[batch_pairs[0]]
        linear_projections = linear_proj(starting_activations, rvlt_directions)

        # opt_act, continue_training, converged, penalties, av_feature_similarities = train_activations_helper(batch, batch_pairs, linear_projections, rvlt_directions, kth_top_loss, final_losses, reg_scheduler, termination=10)
        opt_act, long_batch_pairs, converged, penalties, av_feature_similarities = f_h(batch, batch_pairs, linear_projections, rvlt_directions, kth_top_loss, final_losses, convergence_times, reg_scheduler, initial_losses=initial_losses)

        if long_batch_pairs.shape[1] > 0:
            second_pass.append(long_batch_pairs)

        # longest_convergence = max(longest_convergence, len(converged))
        # agg_converged.append(np.array(converged))
        # agg_penalties.append(np.array(penalties))
        # agg_feature_sim.append(np.array(av_feature_similarities))        
        # if verbose:
        #     plot_curves(converged,penalties,av_feature_similarities)

        #     sns.scatterplot(x=initial_similarities[batch_pairs[0],batch_pairs[1]].cpu().numpy(), y=final_losses[batch_pairs[0],batch_pairs[1]].flatten().cpu().numpy())
        #     plt.show()
        
        kth_top_loss = final_losses.kthvalue(final_losses.shape[1]-top_k, dim=-1)[0]
        optim_activations[batch_pairs[0],batch_pairs[1]] = opt_act.detach()
    
    if len(second_pass) > 0:
        activations_to_train = torch.cat(second_pass, dim=1)
        num_batches = math.ceil((activations_to_train.shape[1]) / features_per_batch)
        for j in tqdm(range(num_batches)):
            start_batch = j * features_per_batch
            end_batch = (j+1) * features_per_batch

            # opt_act = torch.nn.Parameter(cur_activations.unsqueeze(1).repeat(1, features_per_batch, 1))

            batch_pairs = activations_to_train[:,start_batch:end_batch]

            rvlt_directions = feature_directions[batch_pairs[1]]
            starting_activations = optim_activations[batch_pairs[0],batch_pairs[1]]
            linear_projections = linear_proj(starting_activations, rvlt_directions)

            # opt_act, continue_training, converged, penalties, av_feature_similarities = train_activations_helper(batch, batch_pairs, linear_projections, rvlt_directions, kth_top_loss, final_losses, reg_scheduler, termination=10)
            opt_act, longest_pairs, converged, penalties, av_feature_similarities = f_h(batch, batch_pairs, linear_projections, rvlt_directions, kth_top_loss, final_losses, convergence_times, reg_scheduler, start_t=10, soft_limit=500, hard_limit=1500)

            if longest_pairs.shape[1] > 0:
                print(longest_pairs)
                
            longest_convergence = max(longest_convergence, len(converged))
            agg_converged.append(np.array(converged))
            agg_penalties.append(np.array(penalties))
            agg_feature_sim.append(np.array(av_feature_similarities))        
            # if verbose:
            #     plot_curves(converged,penalties,av_feature_similarities)

            #     sns.scatterplot(x=initial_similarities[batch_pairs[0],batch_pairs[1]].cpu().numpy(), y=final_losses[batch_pairs[0],batch_pairs[1]].flatten().cpu().numpy())
            #     plt.show()
            
            kth_top_loss = final_losses.kthvalue(final_losses.shape[1]-top_k, dim=-1)[0]
            optim_activations[batch_pairs[0],batch_pairs[1]] = opt_act.detach()
    
    top_k_losses, idxes = final_losses.topk(top_k, dim=-1)
    batch_feature_idxes = torch.stack([torch.arange(batch_size).unsqueeze(1).repeat(1,top_k).to(device),idxes]).permute(1,2,0).flatten(0,1)
    avg_e_scores = torch.zeros(final_losses.shape).to(device)
    avg_e_scores[batch_feature_idxes[:,0],batch_feature_idxes[:,1]] = top_k_losses.flatten()
    if longest_convergence > 8:
        convergence_graph = []
        for x in [agg_converged, agg_penalties, agg_feature_sim]:
            convergence_graph.append(np.vstack([np.pad(y, (0, longest_convergence - y.shape[0]), 'constant') for y in x]).sum(axis=0)[5:])
    else:
        convergence_graph = None

    return batch_feature_idxes, optim_activations[batch_feature_idxes[:,0],batch_feature_idxes[:,1]], initial_similarities, cur_activations.norm(dim=-1, keepdim=True), initial_losses, convergence_times, final_losses, top_k_losses, avg_e_scores.mean(dim=0), (avg_e_scores > 0).sum(dim=0), convergence_graph

# one activation vec for each feature
# potentially very large

# %%
# %%
i = 0
# %%
folder="v3"
all_feature_directions = []
result_cols = ['cos_sim', 'initial_losses', 'final_losses', 'convergence_times', 'initial_similarities']
result_data = []
avg_e_score = []
updates_per_feature = []
for i in [0,100]:
    with open(f"init_sae/{folder}/feature_{i}.pkl", "rb") as f:
        all_feature_directions.append(pickle.load(f))
    with open(f"init_sae/{folder}/updates_{i}.pkl", "rb") as f:
        updates_per_feature.append(pickle.load(f))

    x={}
    result_data.append(x)
    for col in result_cols:
        x[col] = []
    avg_e_score.append(torch.zeros((num_features,)).to(device))

# %%
while i < 1000:
    batch = next(owt_iter)['tokens']

    for j,feature_directions in enumerate(all_feature_directions):

        idxes, optim_activations, initial_similarities, act_norms, initial_losses, convergence_times, final_losses, e_scores, avg_e, update_ct, convergence_graph = train_activations(model, feature_directions, poly_scheduler)

        cos_sim = (initial_similarities / act_norms).flatten().cpu()

        result_data[j]['cos_sim'].append(cos_sim)
        result_data[j]['initial_losses'].append(initial_losses)
        result_data[j]['final_losses'].append(final_losses)
        result_data[j]['convergence_times'].append(convergence_times)
        result_data[j]['initial_similarities'].append(initial_similarities)
        avg_e_score[j] = (i * avg_e_score[j] + avg_e) / (i+1)
        updates_per_feature[j] += update_ct
    i += 1

    # kl_loss(abl_logits, target_logits)

# %%

with open("init_sae/batched_inference.pkl", "wb") as f:
    pickle.dump([result_data, avg_e_score, updates_per_feature], f)

# %%


for j,res in enumerate(result_data):
    for key in res:
        if isinstance(result_data[j][key], list):
            result_data[j][key] = torch.stack(result_data[j][key], dim=0).flatten()
        else:
            result_data[j][key] = result_data[j][key].flatten()

# %%
with open("init_sae/batched_inference.pkl", "rb") as f:
    [result_data, avg_e_score, updates_per_feature] = pickle.load(f)

# %%

for j,res in enumerate(result_data):
    cos_sim = torch.cat(result_data[j]['cos_sim'], dim=0)
    initial_losses = torch.cat(result_data[j]['initial_losses'], dim=0)
    final_losses = torch.cat(result_data[j]['final_losses'], dim=0)
    convergence_times = torch.cat(result_data[j]['convergence_times'], dim=0)
    initial_similarities = torch.cat(result_data[j]['initial_similarities'], dim=0)
    
    sns.scatterplot(x=cos_sim,y=initial_losses.flatten().cpu(), s=5)
    sns.scatterplot(x=cos_sim,y=final_losses.flatten().cpu(), s=5)
    plt.savefig(f"init_sae/graphs/cos_loss_{j}.png")
    plt.close()
    sns.scatterplot(x=cos_sim,y=final_losses.flatten().cpu(), s=5)
    plt.savefig(f"init_sae/graphs/cos_f_loss_{j}.png")
    plt.close()

    sns.scatterplot(x=initial_losses.flatten().cpu(),y=final_losses.flatten().cpu(), s=5)
    plt.savefig(f"init_sae/graphs/init_final_{j}.png")
    plt.close()

    sns.scatterplot(x=cos_sim,y=convergence_times.flatten().cpu(), s=5)
    plt.savefig(f"init_sae/graphs/cos_conv_{j}.png")
    plt.close()

    sns.scatterplot(x=initial_similarities.flatten().cpu(),y=initial_losses.flatten().cpu(), s=5)
    sns.scatterplot(x=initial_similarities.flatten().cpu(),y=final_losses.flatten().cpu(), s=5)
    plt.savefig(f"init_sae/graphs/proj_loss_{j}.png")
    plt.close()

    sns.scatterplot(x=initial_similarities.flatten().cpu(),y=final_losses.flatten().cpu(), s=5)
    plt.savefig(f"init_sae/graphs/proj_f_loss_{j}.png")
    plt.close()

    sns.scatterplot(x=initial_similarities.flatten().cpu(),y=convergence_times.flatten().cpu(), s=5)
    plt.savefig(f"init_sae/graphs/proj_conv_{j}.png")
    plt.close()

# %%
for j,res in enumerate(result_data):
    updated_features = all_feature_directions[j][(updates_per_feature[j] > 1).nonzero()].squeeze()
    feat_cos_sim = einsum("feat_1 d_model, feat_2 d_model -> feat_1 feat_2", updated_features, updated_features).cpu() - torch.eye(updated_features.shape[0])
    sns.histplot(x=feat_cos_sim.cpu().flatten(),label=f"{j} train")
    print(updated_features.shape)
plt.legend()
plt.savefig(f"init_sae/graphs/feat_sim.png")
plt.close()

# %%
for j,res in enumerate(result_data):
    updated_features = all_feature_directions[j][(updates_per_feature[j] > 5).nonzero()].squeeze()
    feat_cos_sim = (einsum("feat_1 d_model, feat_2 d_model -> feat_1 feat_2", updated_features, updated_features).cpu() - torch.eye(updated_features.shape[0]))
    with open(f"init_sae/v3/feat_sum_{j}.pkl", "wb") as f:
        pickle.dump(feat_cos_sim, f)
    if (feat_cos_sim > .2).nonzero().shape[0] > 0:
        sns.histplot(x=feat_cos_sim.flatten()[(feat_cos_sim.flatten() > .2).nonzero()[:,0]].cpu(), label=f"{j} train")
    print((feat_cos_sim > .6).nonzero()[:,0].unique().shape)
plt.legend()
plt.savefig(f"init_sae/graphs/feat_sim_zoomed.png")
plt.close()
# %%

sns.histplot(updates_per_feature[1][updates_per_feature[1] > 5].flatten().cpu().numpy())

# %%
change_distances = (all_feature_directions[1]- all_feature_directions[0]).square().sum(dim=-1).sqrt()
sns.histplot((change_distances / updates_per_feature[1]).cpu().numpy())

# %%
sns.histplot((change_distances / updates_per_feature[1]).cpu().numpy())
# %%
