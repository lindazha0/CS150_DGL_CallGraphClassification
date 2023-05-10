import os, torch, pickle
import numpy as np
from CallGraphDataset import CallGraphDataset
from experiments import load_dataset, DATA_DIR
from direct_graph_edit_dis import graph_edit_distance

LABELS = os.path.join(DATA_DIR, "5klabels.pt")

def main():
    data = load_dataset(num_graphs=10000)
    len_labels = len(data) // 2
    print(f"dataset size: {len(data)} with {len_labels} pairs")

    labels = []
    for i in range(len_labels):
        g1 = data[i*2]
        g2 = data[i*2+1]
        dist = graph_edit_distance(g1, g2)
        print(f"Graph {g1} and Graph {g2} have edit distance {dist}")
        labels.append(dist)
    torch.save(torch.tensor(labels, dtype=torch.float32), LABELS)


if __name__ == "__main__":
    main()