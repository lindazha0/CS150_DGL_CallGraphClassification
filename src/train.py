import numpy as np
from torch_geometric.loader import DataLoader
import torch
import copy

from model import GNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from direct_graph_edit_dis import graph_edit_distance
from earth_move_dis import emd_distance

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
    return torch.tensor(res)

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
            res.append(emd_distance(embeddings[i].detach().numpy(), embeddings[j].detach().numpy()))
    return torch.tensor(res)

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
    model = GNN(num_features=1, 
            out_dim=20, 
            hid_dim=64, 
            num_layers=5, layer_type='GCNConv')
    trainset, valset = train_test_split(trainset, test_size=0.2)

    # use a loalder to form training batches
    train_loader = DataLoader(trainset, batch_size=2, shuffle=False)

    # training loop to train a model 
    min_val_err = 100
    best_model = None
    epochs = 50
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
            print(f"embeddings: {embeddings.shape}")
            similar_mat = embedding_similarities(embeddings)
            print(f"embeddings similarities: {similar_mat.shape}")
            ground_truth = graph_similarities(batch.to_data_list())
            print(f"ground truth similarities: {ground_truth.shape}")

            # gradient descent
            loss = criterion(similar_mat, ground_truth)
            print(f"loss: {loss}")
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

    return best_model, min_val_err

