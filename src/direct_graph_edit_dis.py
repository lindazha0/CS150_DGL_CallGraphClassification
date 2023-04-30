# import graphsim as gs
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.graph_edit_distance.html
# nx function might not fit directed graphs
import networkx as nx

def graph_edit_distance(G1, G2):
    return nx.graph_edit_distance(G1, G2)

def main():
    # Create directed graphs using NetworkX
    G1 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    G2 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (1, 4)])


    # Compute graph edit distance
    ged = nx.graph_edit_distance(G1, G2)
    print("Graph Edit Distance:", ged)

if __name__ == "__main__":
    main()