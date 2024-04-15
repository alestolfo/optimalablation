# %%

from training_utils import LinePlot
import pickle
from glob import glob
# %%
g = glob("pruning_edges_auto/ioi_iter/*")
# %%
fg = glob(f"{g[0]}/*")
# %%
with open(f"{g[0]}/metadata.pkl", "rb") as f:
    data = pickle.load(f)
# %%
main_log, lp_count, temp_params = data

# %%

main_log.stat_sig_growth("complexity_loss", avg_intv=6, comp_intv=30)

# %%

decls = [stat_sig_growth(main_log.stat_book["complexity_loss"][:x], avg_intv=6, comp_intv=30)[0] for x in range(0,10000,10)]
# %%
s = 400
ed = 490
main_log.plot(["complexity_loss","temp"], start=s, end=ed)
decls[s // 10:ed // 10]
# %%
stat_sig_growth(main_log.stat_book["complexity_loss"][:300], avg_intv=6, comp_intv=30)[0]

# %%
