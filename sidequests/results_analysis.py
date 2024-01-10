# %%

import torch
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from fancy_einsum import einsum
# %%
def load_features(i, folder="v3"):
    with open(f"outputs/{folder}/feature_{i}.pkl", "rb") as f:
        features = pickle.load(f)
    with open(f"outputs/{folder}/updates_{i}.pkl", "rb") as f:
        updates_per_feature = pickle.load(f)
    with open(f"outputs/{folder}/av_e_{i}.pkl", "rb") as f:
        avg_e_score = pickle.load(f)
    return features, updates_per_feature, avg_e_score

# %%

features, updates_per_feature, avg_e_score = load_features(400)
orig_features, _, _ = load_features(0)

# %%
updated_features = features[updates_per_feature.nonzero()].squeeze()
cos_sim = einsum("feat_1 d_model, feat_2 d_model -> feat_1 feat_2", updated_features, updated_features).cpu() - torch.eye(updated_features.shape[0])
orig_cos_sim = einsum("feat_1 d_model, feat_2 d_model -> feat_1 feat_2", orig_features, orig_features).cpu() - torch.eye(orig_features.shape[0])

# %%

sns.histplot(cos_sim.flatten())
sns.histplot(orig_cos_sim.flatten())


# %%
sns.histplot(updates_per_feature[updates_per_feature.nonzero()].cpu().numpy())

# %%
# do PCA first?

# %%
changed_features = (features - orig_features).nonzero()[:,0].unique()
sns.histplot((features-orig_features)[changed_features].norm(dim=-1).cpu().numpy())

# %%



# %%
sns.histplot((features.norm(dim=-1)).cpu().numpy())


# %%
sns.histplot()

# %%
