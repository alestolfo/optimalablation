# %%
import networkx as nx 
import pickle
import matplotlib.pyplot as plt

# %%


# %%
for k in range(6):
    with open(f"outputs/v3/feat_sum_{k}.pkl", "rb") as f:
        feat_cos_sim = pickle.load(f)

    g = nx.DiGraph() 
    for i in range(feat_cos_sim.shape[0]):
        for j in range(i):
            if feat_cos_sim[i][j] > .2:
                g.add_edge(i, j, weight=feat_cos_sim[i][j] ** 2)

    plt.figure(figsize=(10, 10))  
    pos = nx.spring_layout(g, k=.1, iterations=50)
    nx.draw_networkx_nodes(g,pos,node_size=50)

    for edge in g.edges(data='weight'):
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=edge[2])

    plt.savefig(f"outputs/graphs/adjacency_{k}.png")
    plt.close()

# %%
