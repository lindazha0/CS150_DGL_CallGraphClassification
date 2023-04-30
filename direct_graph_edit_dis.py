import graphsim as gs
import networkx as nx

# Create directed graphs using NetworkX
G1 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
G2 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (1, 4)])


# Compute graph edit distance
ged = nx.graph_edit_distance(G1, G2)
print("Graph Edit Distance:", ged)
