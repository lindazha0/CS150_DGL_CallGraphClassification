from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from train import train
import numpy as np
import pickle
from CallGraphDataset import CallGraphDataset
import os

DATASET = "smallDataset.pkl"

def main():
    # load the dataset
    PATH = "../data/preloaded/"
    DATA_PATH = os.path.join(PATH, DATASET)
    if not os.path.exists(DATA_PATH):
        print(f"{DATASET} not loaded, loading from preprocessed call graphs...")
        dataset = CallGraphDataset()
        with open(DATA_PATH, "wb") as f:
            pickle.dump(dataset, f)
        print(f"{DATASET} saved")
    else:
        print(f"{DATASET} loaded")
        with open(DATA_PATH, "rb") as f:
            dataset = pickle.load(f)

    # split the dataset into trainset and testset
    trainset, testset = train_test_split(dataset, train_size=0.8)
    print(f"Total dataset size: {len(dataset)},\ntrainset size: {len(trainset)},\ntestset size: {len(testset)}")

    # learn an optimal model
    print("...Training...")
    model = train(trainset)

    # bellow call model is called to predict test labels 
    print("...Testing...")
    model.eval()
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    for batch in test_loader:
        pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)

    # # Save predictions to the .txt file
    # save_file = "gnn_predictions.txt"
    # print("%d model predictions saved to %s" % (pred.shape[0], save_file))
    # np.savetxt(save_file, pred)

if __name__ == "__main__":
    main()