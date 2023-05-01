# for running experiments on the dataset
import os, torch, pickle
import numpy as np
from CallGraphDataset import CallGraphDataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from train import train, graph_similarities, embedding_similarities
from sklearn.metrics import f1_score
from model import GNN


DATASET_NAME = "1kDataset.pkl"
MODEL_NAME = "1kModel.pt"

DATA_PATH = "../data/preloaded/"
MODEL_PATH = "../models/"

def main():
    # load the dataset
    DATASET = os.path.join(DATA_PATH, DATASET_NAME)
    if not os.path.exists(DATASET):
        print(f"{DATASET} not existed, constructing from preprocessed call graphs...")
        try:
            dataset = CallGraphDataset(1, 1000) # 1 file, 1000 graphs
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

    # load or learn an optimal model
    model = GNN(num_features=1, 
            out_dim=20, 
            hid_dim=64, 
            num_layers=5, layer_type='GCNConv')
    MODEL = os.path.join(MODEL_PATH, MODEL_NAME)
    if os.path.exists(MODEL):
        print(f"{MODEL} existed, loading model...")
        model.load_state_dict(torch.load(MODEL))
    else:
        print("...Training...")
        model, val_err = train(trainset, model)

    # call model to predict test labels 
    print("...Testing...")
    model.eval()
    similarity_threshold = 0.8
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    scores = []
    for batch in test_loader:
        embeddings = model(batch.x, batch.edge_index, batch.batch)
        ground_truth = graph_similarities(batch.to_data_list())
        similar_mat = embedding_similarities(embeddings)

        # Convert similarity matrices into binary labels based on the threshold
        ground_truth_labels = (ground_truth >= similarity_threshold).astype(int)
        predicted_labels = (similar_mat >= similarity_threshold).astype(int)

        # Calculate the F1 score as a evaluation metric
        scores.append(f1_score(ground_truth_labels.flatten(), predicted_labels.flatten()))
    print(f"F1 score for testing: {np.mean(scores)}")

    # Save predictions to the .txt file
    if not os.path.exists(MODEL):
        print(f"saving model...")
        torch.save(model.state_dict(), MODEL)
        print(f"{MODEL} saved")

if __name__ == "__main__":
    main()