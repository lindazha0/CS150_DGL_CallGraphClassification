import torch
import copy, os

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from direct_graph_edit_dis import graph_edit_distance
from earth_move_dis import sliced_wasserstein_distance

# the allowed threshold for distance difference between two graphs to be considered similar
ERR_THRESHOLD = 0.5

def graph_similarities(graphs):
    """
    Compute graph similarities between all pairs of graphs, return a similarity matrix
    args:
        graphs: a list of graphs
    """
    res = []
    for i in range(len(graphs)):
        for j in range(len(graphs)):
            if i == j:
                continue
            dist = graph_edit_distance(graphs[i], graphs[j])
            print(f"Graph {i} and Graph {j} have edit distance {dist}")
            res.append(dist)
    return torch.tensor(res, dtype = torch.float32)

def embedding_similarities(embeddings):
    """
    Compute embedding similarities between all pairs of embeddings, return a similarity matrix
    args:
        embeddings: a list of embeddings
    """
    res = []
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i == j:
                continue
            # calculate the EMD between two embeddings and add a dimension for concatenation
            dist = sliced_wasserstein_distance(embeddings[i], embeddings[j]).unsqueeze(0)
            # print(f"Embedding {i} and Embedding {j} have EMD {dist}")
            res.append(dist)
    return torch.cat(res, dim=0)

def validate(model, valset, val_labels):
    """
    Validate the model by checking its accuracy on the validation set of nodes
    args:
        model: a GNN object 
        testset: a dataset
    """
    # set the model to the evaluation mode 
    model.eval()

    res = []
    for i in range(len(val_labels)):
        ground_truth = val_labels[i]
        if ground_truth <= 0:
            continue
        g1, g2 = valset[i*2], valset[i*2+1]
        embed_g1, embed_g2 = model(g1.x, g1.edge_index), model(g2.x, g2.edge_index)
        similar_of_emb = sliced_wasserstein_distance(embed_g1, embed_g2)
        # print(f"vali-{i}: G1 {g1}, G2 {g2}, GED {ground_truth}, EMD {similar_of_emb}")
        error = abs(similar_of_emb - ground_truth)/ground_truth
        # print('{:.8f}'.format(error), end=' ')
        res.append(error <= ERR_THRESHOLD)
    # print('\n')
    accuracy = sum(res) / len(res)

    return accuracy

def train(model, trainset, train_labels):
    """
    A training function that trains a GNN model 

    args:
        trainset: a dataset
    """
    # split the dataset into training and validation sets
    trainset, valset = train_test_split(trainset, test_size=0.2, shuffle=False)
    train_y, val_y = train_test_split(train_labels, test_size=0.2, shuffle=False)

    # training loop to train a model 
    max_val_acc = 0.0
    best_model = model
    epochs = 50
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("before training:", model)

    # train the model
    for epoch in range(epochs):
        # train the model
        for i in range(len(train_y)):
            if train_y[i] <= 0:
                continue
            model.train()

            # clear the gradients
            optimizer.zero_grad()

            # forward pass
            g1, g2 = trainset[i*2], trainset[i*2+1]
            embeddings_g1 = model(g1.x, g1.edge_index)
            embeddings_g2 = model(g2.x, g2.edge_index)
            # print(f"embeddings: {embeddings_g1.shape}")
            similar_of_emb = sliced_wasserstein_distance(embeddings_g1, embeddings_g2)
            # print(f"embeddings similarities: {similar_of_emb}, {similar_of_emb.dtype}")
            ground_truth = train_y[i]
            # print(f"ground truth similarities: {ground_truth}, {ground_truth.dtype}")

            # gradient descent
            loss = criterion(similar_of_emb, ground_truth)
            loss.backward()
            optimizer.step()

        # print the loss for each epoch
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss.item()))

        # validate the model
        val_acc = validate(model, valset, val_y)

        # save the best model
        if val_acc >= max_val_acc:
            max_val_acc = val_acc
            best_model = copy.deepcopy(model)
    print('Finished training!')
    print(f"Best model: {best_model}, is last one: {best_model==model}")
    print('Best validation accuracy: {:.5f}'.format(max_val_acc))

    return best_model, max_val_acc

