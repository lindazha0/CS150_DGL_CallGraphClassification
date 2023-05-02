import torch, os
import experiments as exp
from direct_graph_edit_dis import graph_edit_distance

TRAIN_LABELS = "1kTrainLabels.pt"
TEST_LABELS = "1kTestLabels.pt"

def main():
    trainset, testset = exp.load_train_test_set()
    train_len, test_len = len(trainset) // 2, len(testset) // 2
    print(type(trainset), type(testset), type(trainset[0]))
    print(f"trainset size: {train_len} pairs,\ntestset size: {test_len} pairs")

    # calculate graph edit distance for trainset
    train_labels = []
    for i in range(train_len):
        g1 = trainset[i*2]
        g2 = trainset[i*2+1]
        dist = graph_edit_distance(g1, g2)
        print(f"Graph {g1} and Graph {g2} have edit distance {dist}")
        train_labels.append(dist)
    torch.save(torch.tensor(train_labels, dtype=torch.float32), os.path.join(exp.DATA_DIR, TRAIN_LABELS))

    # calculate graph edit distance for testset
    test_labels = []
    for i in range(test_len):
        g1 = testset[i*2]
        g2 = testset[i*2+1]
        dist = graph_edit_distance(g1, g2)
        print(f"Graph {g1} and Graph {g2} have edit distance {dist}")
        test_labels.append(dist)
    torch.save(torch.tensor(test_labels, dtype=torch.float32), os.path.join(exp.DATA_DIR, TEST_LABELS))

if __name__ == "__main__":
    main()