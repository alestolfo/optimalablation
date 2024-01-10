# %%

from glob import glob
# %%
graphs = sorted(glob("outputs/graphs/*"))

for g in graphs:
    print(f"\includegraphics[width=4in]{{{g.split('/')}}}")
# %%
