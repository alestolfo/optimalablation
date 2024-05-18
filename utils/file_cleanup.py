
# %%

import glob
import os
import torch
# %%

for x in glob.glob("../results/pruning/ioi/mean/unif-old/*"):
    print(x)
    try:
        float(x.split("/")[-1])
    except:
        continue
    for y in glob.glob(f"{x}/*"):
        if y.endswith("9.png") or y.endswith("9.pkl") or y.endswith("9.pth") or y.endswith("final.pkl") or y.endswith("final.png") or y.endswith("final.pth"):
            os.remove(y)
        # elif y.endswith("snapshot.pth"):
        #     snapshot = torch.load(y)
        #     keys = snapshot['pruner_dict'].keys()
        #     reset = False
        #     for k in keys:
        #         if k.startswith("base_model"):
        #             print("ahh")
        #             reset = True
        #     if reset:
        #         print(f"NEED TO RESET {y}")
        #         new_dict = {k: snapshot['pruner_dict'][k] for k in snapshot['pruner_dict'] if not k.startswith("base_model")}
        #         snapshot['pruner_dict'] = new_dict
        #         torch.save(y, snapshot)
                
# %%
