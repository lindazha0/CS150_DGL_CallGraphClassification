import os
import misc as c
from collections import defaultdict

def main():
    # get list of graphs from pkl files
    files = [e for e in os.scandir(os.path.join(c.DATA_FOLDER, c.GRAPHS_V2)) if e.is_file() and e.name.endswith('.pkl')]

    # loop over each pkl file as graphs
    num_files = 1
    services = defaultdict(list)
    
    for f in files[:num_files]:
        graphs = c.read_result_object(f.path) # [tracedataList, edgeList, edgefeatures]
        print(type(graphs))
        print(len(graphs))

        # read graphs[0]
        graph = graphs[0]
        print(type(graph))
        print(graph)
        print(graph.edgelist)
        print(graph.nodefeatures)
        print(graph.trace)

    return

if __name__ == "__main__":
    main()