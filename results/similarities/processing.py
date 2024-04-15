# %%

import pandas as pd
import numpy as np
# %%

for y in ["ioi","gt"]:
    df = pd.read_csv(f"{y}/total_nodes.csv")
    df['method'] = df['key'].str.split('/').str[:-1].str.join('/')
    df['lamb'] = df['key'].str.split('/').str[-1]

    df.loc[(df['method'] == f"acdc_{y}_runs") & (df['lamb'] == "manual"), ['method','lamb']] = ["0_manual", 0]
    df['lamb'] = df['lamb'].str.split("-0.0").str[0]
    df['lamb'] = df['lamb'].astype(float)

    summary_df = df.sort_values(["method","lamb"], ascending=[True, False])[['method','lamb','nodes','edges','loss']]

    summary_df.to_csv(f"{y}/total_nodes_updated.csv")

    for x in ["edge","node"]:
        df = pd.read_csv(f"{y}/{x}_similarities.csv")

        df['method'] = df['key1'].str.split('/').str[:-1].str.join('/')
        df['lamb'] = df['key1'].str.split('/').str[-1]

        df.loc[(df['method'] == f"acdc_{y}_runs") & (df['lamb'] == "manual"), ['method','lamb']] = ["0_manual", 0]
        df['lamb'] = df['lamb'].str.split("-0.0").str[0]
        df['lamb'] = df['lamb'].astype(float)

        df = df.sort_values(["method","lamb"], ascending=[True, False])

        df = df.merge(summary_df, how="left", on=["method","lamb"])

        final_edge_df = pd.DataFrame(np.concatenate([df.values[:,-5:], df.values[:,df['Unnamed: 0']+2]],axis=1))
        final_edge_df.to_csv(f"{y}/{x}_similarities_updated.csv")


# %%
