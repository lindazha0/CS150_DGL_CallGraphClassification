import os
from collections import defaultdict
import pickle

# for pytorch geometric
import torch
from torch_geometric.data import Data


PKL_FOLDER = os.path.join(os.path.join('..', 'data'), 'graphs_v2')

def to_pyg_graph(graph):
    """
    Construct a PyG graph from a call graph object
    args:
        graph : a call graph object, with the following attributes:
            - graph.edgelist: a list of edges, each edge is a tuple of two nodes
            - graph.nodefeatures: microservice index within the whole trace dataset
            - graph.trace
    """
    # Convert to tensors
    node_features = torch.tensor(graph.nodefeatures, dtype=torch.float).view(-1, 1) # Reshape to (num_nodes, 1)
    edge_list = torch.tensor(graph.edgelist, dtype=torch.long).view(2, -1) # Reshape to (2, num_edges)

    # Create a PyG graph with edge_index and edge_attr
    graph = Data(x=node_features, edge_index=edge_list)
    return graph

def call_graph_dataset(num_files=1, num_graphs=1000, describe=False):
    """
    Load call graphs from pkl files as a list of pyg graphs
    args:
        num_files: number of files to load (default: 1)
    """
    # get list of graphs from pkl files
    files = [e for e in os.scandir(PKL_FOLDER) if e.is_file() and e.name.endswith('.pkl')]

    # loop over each pkl file as graphs
    graph_list = []
    if num_files == 1:
        files = [files[5]]
    for f in files[:num_files]:
        # form: [tracedataList, edgeList, edgefeatures]
        graphs = pickle.load(open(f.path, "rb"))

        # for debugging
        if describe:
            print(f"Reading graphs from {f.name}")
            print(type(graphs))
            print(len(graphs))

                # read graphs[0]
            graph = graphs[0]
            print(f"Reading first graph, with {len(graph.nodefeatures)} nodes and {len(graph.edgelist)} edges")
            print(type(graph))
            print(graph)
            print(graph.edgelist)
            print(graph.nodefeatures)
            print(graph.trace)

        num_graphs = min(num_graphs, len(graphs))
        for g in graphs[:num_graphs]:
            # print(f"Reading graph with {len(g.nodefeatures)} nodes and {len(g.edgelist)} edges")
            # convert to pyg graph
            pyg_graph = to_pyg_graph(g)
            graph_list.append(pyg_graph)
    return graph_list

def main():
    # get list of graphs from pkl files
    graph_list = call_graph_dataset(num_files=1, num_graphs=10000, describe=True)

    return

if __name__ == "__main__":
    main()