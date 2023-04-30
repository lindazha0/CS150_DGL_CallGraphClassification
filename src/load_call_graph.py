import os
import misc as c
from collections import defaultdict

# for pytorch geometric
import torch
from torch_geometric.data import Data

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
    edge_index = torch.tensor(graph.edgelist, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(graph.nodefeatures, dtype=torch.float32).view(-1, 1)

    # Create a PyG graph with edge_index and edge_attr
    graph = Data(node_features=node_features, edge_index=edge_index)
    return graph

def call_graph_dataset(num_files=1):
    """
    Load call graphs from pkl files as a list of pyg graphs
    args:
        num_files: number of files to load (default: 1)
    """
    # get list of graphs from pkl files
    files = [e for e in os.scandir(os.path.join(c.DATA_FOLDER, c.GRAPHS_V2)) if e.is_file() and e.name.endswith('.pkl')]

    # loop over each pkl file as graphs
    graph_list = []
    for f in files[:num_files]:
        graphs = c.read_result_object(f.path) # [tracedataList, edgeList, edgefeatures]
        for g in graphs:
            # convert to pyg graph
            pyg_graph = to_pyg_graph(g)
            graph_list.append(pyg_graph)
    return graph_list

def main():
    # get list of graphs from pkl files
    files = [e for e in os.scandir(os.path.join(c.DATA_FOLDER, c.GRAPHS_V2)) if e.is_file() and e.name.endswith('.pkl')]

    # loop over each pkl file as graphs
    num_files = 1
    
    for f in files[:num_files]:
        print(f"Reading graphs from {f.name}")
        graphs = c.read_result_object(f.path) # [tracedataList, edgeList, edgefeatures]
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

    return

if __name__ == "__main__":
    main()