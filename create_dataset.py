# %%

from huggingface_hub import create_repo, login,HfApi
from datasets import load_dataset

# %%
login('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')
# %%
create_repo("maxtli/OpenWebText-2M", repo_type="dataset")
# %%
api = HfApi()

# %%
api.upload_folder(
    folder_path="SAE_training/owt_25p",
    repo_id="maxtli/OpenWebText-2M",
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
)



# %%

load_dataset("maxtli/OpenWebText-2M")
# %%
