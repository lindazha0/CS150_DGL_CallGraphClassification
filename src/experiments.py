from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from train import train
import numpy as np
import pickle
from CallGraphDataset import CallGraphDataset
import os
import torch

DATASET_NAME = "smallDataset.pkl"
MODEL_NAME = "smallModel.pt"

DATA_PATH = "../data/preloaded/"
MODEL_PATH = "../models/"

def main():
    # load the dataset
    DATASET = os.path.join(DATA_PATH, DATASET_NAME)
    if not os.path.exists(DATASET):
        print(f"{DATASET} not loaded, loading from preprocessed call graphs...")
        try:
            dataset = CallGraphDataset()
        except:
            print("Failed to load the dataset")
            return
        with open(DATASET, "wb") as f:
            pickle.dump(dataset, f)
        print(f"{DATASET} saved")
    else:
        print(f"{DATASET} loaded")
        with open(DATASET, "rb") as f:
            dataset = pickle.load(f)

    # split the dataset into trainset and testset
    trainset, testset = train_test_split(dataset, train_size=0.8)
    print(f"Total dataset size: {len(dataset)},\ntrainset size: {len(trainset)},\ntestset size: {len(testset)}")

    # learn an optimal model
    print("...Training...")
    model, val_err = train(trainset)

    # bellow call model is called to predict test labels 
    print("...Testing...")
    # model.eval()
    # test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    # for batch in test_loader:
    #     embeddings = model(batch.x, batch.edge_index, batch.batch)
    #     ground_truth = graph_similarities(batch.to_data_list())
    #     similar_mat = embedding_similarities(embeddings)
    #     err = mean_squared_error(similar_mat, ground_truth)

    # Save predictions to the .txt file
    MODEL = os.path.join(MODEL_PATH, MODEL_NAME)
    if not os.path.exists(MODEL):
        print(f"{MODEL} not exists, saving model...")
        torch.save(model.state_dict(), MODEL)
        print(f"{MODEL} saved")

if __name__ == "__main__":
    main()