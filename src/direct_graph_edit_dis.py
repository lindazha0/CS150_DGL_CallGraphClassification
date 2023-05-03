# import graphsim as gs
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.graph_edit_distance.html
# nx function might not fit directed graphs
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch, time, csv
from timer import time_limit, TimeoutException

GED_TIME_FILE = 'times_for_GED_srun10_preempt.csv'
GED_TIME_LIMIT = 10 * 60 # unit: seconds

def custom_node_match(node1, node2):
    """
    Define a custom node matcher for graph edit distance
    """
    # print(f"node1: {node1}")
    feature1 = node1['x']
    feature2 = node2['x']

    return np.isclose(feature1, feature2, atol=0.1)

def graph_edit_distance(G1, G2):
    """
    Compute graph edit distance between two graphs. Their node features, as a single value, would be taken into consideration.
    args:
        G1, G2: two graphs in the form of pyg Data
    """
    begin = time.time()
    nx_G1 = to_networkx(G1, node_attrs = ['x'], to_undirected=False)
    nx_G2 = to_networkx(G2, node_attrs = ['x'], to_undirected=False)
    print(f"calculate dist for graphs with {nx_G1.number_of_nodes()} and {nx_G2.number_of_nodes()} nodes")

    # for v in nx.optimize_graph_edit_distance(nx_G1, nx_G2, node_match=custom_node_match):
        # ged = v
    ged = -1
    try:
        with time_limit(GED_TIME_LIMIT):
            ged = nx.graph_edit_distance(nx_G1, nx_G2, node_match=custom_node_match)
    except TimeoutException as e:
        print(f"Graph Edit Distance Timed Out!")

    # record time elapsed
    elapsed = time.time() - begin
    elapsed_format =  time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    with open(GED_TIME_FILE, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([nx_G1.number_of_nodes(), nx_G2.number_of_nodes(),elapsed, elapsed_format, ged])
    print(f"Time elapsed: {elapsed_format}")

    return ged

def main():
    # Create directed graphs using NetworkX
    x = torch.tensor([7, 2, 55, 1, 5, 6])
    G = Data(x=x, edge_index=torch.tensor([[0, 1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 1, 0]]))
    G3 = Data(x=torch.tensor([3,4,5,6,2,4,6,1,44,5,6,2,6,77,5,2]), edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]]) )
    G1 = to_networkx(G, node_attrs = ['x'])

    G2 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 1), (3, 0)])
    for i, feature in enumerate(G.x):
        G2.nodes[i]['x'] = feature.item()

    # Compute graph edit distance
    # print("Graph 1:", G1.nodes.data())
    ged = graph_edit_distance(G, G3)
    print("Graph Edit Distance:", ged, "of type", type(ged))

if __name__ == "__main__":
    main()