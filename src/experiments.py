# for running experiments on the dataset
import os, torch, pickle
import numpy as np
from CallGraphDataset import CallGraphDataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from train import train, graph_similarities, sliced_wasserstein_distance, VALID_THRESHOLD
from sklearn.metrics import f1_score
from model import GNN

TRAIN = True
DATASET_NAME = "10Dataset.pkl"

TRAINSET_NAME = "1kTrainset.pkl"
TESTSET_NAME = "10Testset.pkl"
TRAIN_LABELS = "50TrainLabels.pt"
TEST_LABELS = "50TrainLabels.pt"
# TEST_LABELS = "10TestLabels.pt"

MODEL_PT_NAME = "50Model.pt"

DATA_DIR = "../data/preloaded/"
MODEL_PT_PATH = "../models/"


def load_dataset(num_files=1, num_graphs=10):
    """
    Load the dataset from the preprocessed files
    args:
        file_num: the number of files to load
        num_graphs: the number of graphs to load from each file
    """
    DATASET = os.path.join(DATA_DIR, DATASET_NAME)
    TRAINSET = os.path.join(DATA_DIR, TRAINSET_NAME)
    TESTSET = os.path.join(DATA_DIR, TESTSET_NAME)
    if not os.path.exists(DATASET):
        print(f"{DATASET} not existed, constructing from preprocessed call graphs...")
        try:
            dataset = CallGraphDataset(num_files, num_graphs) # 1 file, 1000 graphs
        except Exception:
            print("Failed to load the dataset")
            raise Exception
        with open(DATASET, "wb") as f:
            pickle.dump(dataset, f)
        print(f"{DATASET} saved")
    else:
        print(f"{DATASET} loaded")
        with open(DATASET, "rb") as f:
            dataset = pickle.load(f)

    # split the dataset into trainset and testset
    trainset, testset = train_test_split(dataset, train_size=0.8)
    with open(TRAINSET, "wb") as f:
        pickle.dump(trainset, f)
    with open(TESTSET, "wb") as f:
        pickle.dump(testset, f)

    return trainset, testset

def load_train_test_set():
    """
    Load the trainset and testset from the preprocessed files
    """
    TRAINSET = os.path.join(DATA_DIR, TRAINSET_NAME)
    TESTSET = os.path.join(DATA_DIR, TESTSET_NAME)

    if not os.path.exists(TRAINSET) or not os.path.exists(TESTSET):
        print(f"{TRAINSET} not existed, loading dataset...")
        return load_dataset()
    else:
        print(f"{TRAINSET} loaded")
        with open(TRAINSET, "rb") as f:
            trainset = pickle.load(f)
        print(f"{TESTSET} loaded")
        with open(TESTSET, "rb") as f:
            testset = pickle.load(f)
        return trainset, testset

def load_train_test_labels():
    """
    Load the trainset and testset from the preprocessed files
    """
    if not os.path.exists(os.path.join(DATA_DIR, TRAIN_LABELS)):
        print(f"{TRAIN_LABELS} not existed, generating...")
        main()
    train_labels = torch.load(os.path.join(DATA_DIR, TRAIN_LABELS))
    test_labels = torch.load(os.path.join(DATA_DIR, TEST_LABELS))
    return train_labels, test_labels

def main():
    # load the dataset
    trainset, testset = load_train_test_set()
    train_y, test_y = load_train_test_labels()

    # for unit test of 50 in training set
    trainset, testset = trainset[:80], trainset[80:100]
    train_y, test_y = train_y[:40], test_y[40:]
    print(f"trainset size: {len(trainset)},\ntestset size: {len(testset)}")
    print(f"train_y size: {len(train_y)},\ntest_y size: {len(test_y)}")

    # load or learn an optimal model
    model = GNN(num_features=1, 
            out_dim=20, 
            hid_dim=64, 
            num_layers=5, layer_type='GCNConv')
    MODEL_PT = os.path.join(MODEL_PT_PATH, MODEL_PT_NAME)
    if os.path.exists(MODEL_PT) and not TRAIN:
        print(f"{MODEL_PT} existed, loading model...")
        model.load_state_dict(torch.load(MODEL_PT))
    else:
        print("...Training...")
        model, val_acc = train(model, trainset, train_y)

        # save best model
        print(f"saving model...")
        torch.save(model.state_dict(), MODEL_PT)
        print(f"{MODEL_PT} saved")

    
    # call model to predict test labels 
    print("...Testing...")
    model.eval()
    # test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    scores = []
    # test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
    for i in range(len(test_y)):
        ground_truth = test_y[i]
        if ground_truth <= 0:
            continue
        g1, g2 = testset[i*2], testset[i*2+1]
        embed_g1, embed_g2 = model(g1.x, g1.edge_index), model(g2.x, g2.edge_index)
        similar_of_emb = sliced_wasserstein_distance(embed_g1, embed_g2)
        print(f"vali-{i}: G1 {g1}, G2 {g2}, GED {ground_truth}, EMD {similar_of_emb}")
        scores.append(abs(similar_of_emb - ground_truth) <= VALID_THRESHOLD)
    accuracy = sum(scores) / len(scores)
    print(f"accuracy for testing: {accuracy}")

if __name__ == "__main__":
    main()