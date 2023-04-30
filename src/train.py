import numpy as np
from torch_geometric.loader import DataLoader
import torch
import copy

from model import GNN
from sklearn.model_selection import train_test_split, mean_squared_error
from direct_graph_edit_dis import graph_edit_distance
from earth_move_dis import wasserstein_distance

def graph_similarities(graphs):
    """
    Compute graph similarities between all pairs of graphs, return a similarity matrix
    args:
        graphs: a list of graphs
    """
    res = []
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            if i == j:
                continue
            res.append(graph_edit_distance(graphs[i], graphs[j]))
    return res

def embedding_similarities(embeddings):
    """
    Compute embedding similarities between all pairs of embeddings, return a similarity matrix
    args:
        embeddings: a list of embeddings
    """
    res = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if i == j:
                continue
            res.append(wasserstein_distance(embeddings[i], embeddings[j]))
    return res

def validate(model, testset):
    """
    Validate the model by checking its accuracy on the validation set of nodes
    args:
        model: a GNN object 
        testset: a dataset
    """

    # set the model to the evaluation mode 
    model.eval()

    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
    for batch in test_loader:
        embeddings = model(batch.x, batch.edge_index, batch.batch)
        ground_truth = graph_similarities(batch.to_data_list())
        similar_mat = embedding_similarities(embeddings)
        err = mean_squared_error(similar_mat, ground_truth)

    return err

def train(trainset):
    """
    A training function that trains a GNN model 

    args:
        trainset: a dataset
    """
    # split the dataset into training and validation sets
    model = GNN(num_features=trainset.num_features, 
            out_dim=trainset.num_classes, 
            hid_dim=64, 
            num_layers=4, layer_type='GCNConv')
    trainset, valset = train_test_split(trainset, test_size=0.2)

    # NOTE: please use a loalder to form training batches
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True)

    # training loop to train a model 
    min_val_err = 100
    best_model = None
    epochs = 100
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train the model
    for epoch in range(epochs):
        for batch in train_loader:
            model.train()

            # clear the gradients
            optimizer.zero_grad()

            # forward pass
            embeddings = model(batch.x, batch.edge_index, batch.batch)
            ground_truth = graph_similarities(batch.to_data_list())
            similar_mat = embedding_similarities(embeddings)
            loss = criterion(similar_mat, ground_truth)
            loss.backward()
            optimizer.step()
        # print the loss
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss.item()))

        # validate the model
        val_err = validate(model, valset)

        # save the best model
        if val_err < min_val_err:
            min_val_err = val_err
            best_model = copy.deepcopy(model)
    print('Minimal validation error: {:.5f}'.format(min_val_err))

    return best_model

