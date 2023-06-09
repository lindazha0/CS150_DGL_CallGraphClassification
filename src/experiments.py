# for running experiments on the dataset
import os, torch, pickle
import numpy as np
from CallGraphDataset import CallGraphDataset
from sklearn.model_selection import train_test_split
from train import train, sliced_wasserstein_distance, ERR_THRESHOLD
from model import GNN

TRAIN = True
NUM_LABELS = 400 # specify the number of labels to use, as well as the number of graphs to load
DATASET_NAME = "1kDataset.pkl" # fixed by default

TRAINSET_NAME = "1kTrainset.pkl"
TESTSET_NAME = "1kTestset.pkl"
TRAIN_LABELS = "400_TrainLabels.pt"
# TEST_LABELS = TRAIN_LABELS
TEST_LABELS = "100_TestLabels.pt"

MODEL_TYPE = "GATv2"
MODEL_PT_NAME = f"500_128_{MODEL_TYPE}_Model.pt"

DATA_DIR = "../data/preloaded/"
MODEL_PT_PATH = "../models/"

def load_dataset(num_files=1, num_graphs=1000):
    """
    Load the dataset from the preprocessed files
    args:
        file_num: the number of files to load
        num_graphs: the number of graphs to load from each file
    """
    DATASET = os.path.join(DATA_DIR, DATASET_NAME) if num_graphs==1000 else os.path.join(DATA_DIR, f"{num_graphs//1000}kDataset.pkl")
    if not os.path.exists(DATASET):
        print(f"{DATASET} not existed, constructing from preprocessed call graphs...")
        try:
            dataset = CallGraphDataset(num_files, num_graphs) # 1 file, 1000 graphs
        except Exception:
            print("Failed to load the dataset")
            raise Exception
        with open(DATASET, "wb") as f:
            # save the dataset
            pickle.dump(dataset, f)
        print(f"{DATASET} saved")
    else:
        print(f"{DATASET} loaded")
        with open(DATASET, "rb") as f:
            dataset = pickle.load(f)

    return dataset

def load_train_test_set(num_graphs=1000):
    """
    Load the trainset and testset from the preprocessed files
    """
    TRAINSET = os.path.join(DATA_DIR, TRAINSET_NAME)
    TESTSET = os.path.join(DATA_DIR, TESTSET_NAME)

    if not os.path.exists(TRAINSET) or not os.path.exists(TESTSET):
        print(f"{TRAINSET} not existed, loading dataset...")
        # split the dataset into trainset and testset
        dataset = load_dataset(num_graphs=num_graphs)
        trainset, testset = train_test_split(dataset, train_size=0.8)

        # save the trainset and testset
        with open(TRAINSET, "wb") as f:
            pickle.dump(trainset, f)
        with open(TESTSET, "wb") as f:
            pickle.dump(testset, f)
        return load_dataset()
    else:
        print(f"{TRAINSET} loaded")
        with open(TRAINSET, "rb") as f:
            trainset = pickle.load(f)
        print(f"{TESTSET} loaded")
        with open(TESTSET, "rb") as f:
            testset = pickle.load(f)
    return trainset, testset

def load_train_test_labels(train_name=TRAIN_LABELS, test_name=TEST_LABELS):
    """
    Load the trainset and testset from the preprocessed files
    """
    train_path, test_path = os.path.join(DATA_DIR, train_name), os.path.join(DATA_DIR, test_name)
    if not os.path.exists(os.path.join(DATA_DIR, TRAIN_LABELS)):
        print(f"{TRAIN_LABELS} not existed, generating...")
        main()
    train_labels = torch.load(train_path)
    test_labels = torch.load(test_path)
    return train_labels, test_labels

def main():
    torch.set_printoptions(precision=8)
    # load the dataset
    trainset, testset = load_train_test_set()
    train_y, test_y = load_train_test_labels()
    train_y_ht, test_y_ht = load_train_test_labels(train_name="400_TrainLabels_head_tail.pt", test_name="100_TestLabels_head_tail.pt")

    # for unit test of 50 in training set
    # train_size = int(NUM_LABELS*0.8)
    # trainset, testset = trainset[:train_size*2], trainset[train_size*2:NUM_LABELS*2]
    # train_y, test_y = train_y[:train_size], train_y[train_size:]
    print(f"trainset size: {len(trainset)},\ntestset size: {len(testset)}")
    print(f"train_y size: {len(train_y)},\ntest_y size: {len(test_y)}")
    print(f"train_y_ht size: {len(train_y_ht)},\ntest_y_ht size: {len(test_y_ht)}")

    # load or learn an optimal model
    MODEL_PT = os.path.join(MODEL_PT_PATH, MODEL_PT_NAME)
    model = GNN(num_features=1, 
            out_dim=20, 
            hid_dim=128, 
            num_layers=5, layer_type=f'{MODEL_TYPE}Conv')
    if os.path.exists(MODEL_PT):
        print(f"{MODEL_PT} existed, loading model...")
        model.load_state_dict(torch.load(MODEL_PT))
    if TRAIN:
        print("...Training...")
        model, val_acc = train(model, trainset, train_y, train_y_ht)

        # save best model
        print(f"saving model...")
        torch.save(model.state_dict(), MODEL_PT)
        print(f"{MODEL_PT} saved")

    
    # call model to predict test labels 
    print("...Testing...")
    model.eval()
    scores = []
    for i in range(len(test_y)):
        ground_truth = test_y[i]
        if ground_truth <= 0:
            continue
        g1, g2 = testset[i*2], testset[i*2+1]
        embed_g1, embed_g2 = model(g1.x, g1.edge_index), model(g2.x, g2.edge_index)
        similar_of_emb = sliced_wasserstein_distance(embed_g1, embed_g2)
        error = abs(similar_of_emb - ground_truth)/ground_truth
        # print(f"test-{i}: G1 {g1}, G2 {g2}, GED {ground_truth}, EMD {similar_of_emb}, error {error}")
        # print(error)
        scores.append(error <= ERR_THRESHOLD)

    # head and tail
    for i in range(len(test_y_ht)):
        ground_truth = test_y_ht[i]
        if ground_truth <= 0:
            continue
        g1, g2 = testset[i], testset[len(test_y_ht)-1-i]
        embed_g1, embed_g2 = model(g1.x, g1.edge_index), model(g2.x, g2.edge_index)
        similar_of_emb = sliced_wasserstein_distance(embed_g1, embed_g2)
        error = abs(similar_of_emb - ground_truth)/ground_truth
        # print(f"test-{i}: G1 {g1}, G2 {g2}, GED {ground_truth}, EMD {similar_of_emb}, error {error}")
        # print(error)
        scores.append(error <= ERR_THRESHOLD)
    accuracy = sum(scores) / len(scores)
    print(f"accuracy for testing: {accuracy}")

if __name__ == "__main__":
    main()