import numpy as np
from torch_geometric.loader import DataLoader
import torch
import copy

#TODO: please import your gnn from gnn.py
from gnn import GNN
from sklearn.model_selection import train_test_split

#TODO: please implement a function that checks the accuracy of the validation set
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
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        acc = pred.eq(batch.y).sum().item() / batch.y.size(0)

    return acc

#TODO: please implement a function that trains model 
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

    # TODO: please implement a training loop to train a model 
    best_val_acc = 0.0
    best_model = None
    epochs = 100
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # train the model
        for batch in train_loader:
            model.train()

            # clear the gradients
            optimizer.zero_grad()

            # forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        # print the loss
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss.item()))

        # validate the model
        val_acc = validate(model, valset)

        # save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
    print('Best validation accuracy: {:.5f}'.format(best_val_acc))

    return best_model

