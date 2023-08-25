"""
===========
Simple Path
===========

Draw a graph with matplotlib.
"""
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams["font.family"] = "Arial"
G = nx.path_graph(8)
pos = nx.spring_layout(G, seed=47)  # Seed layout for reproducibility
nx.draw(G, pos=pos)
plt.show()
