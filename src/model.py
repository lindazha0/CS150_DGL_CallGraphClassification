import torch
from torch.nn import Linear,functional as F
# NOTE: you can import more convolutional and pooling layers here 
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool


# TODO: implement a GNN 
class GNN(torch.nn.Module):

    # NOTE: please add whatever arguments you want
    def __init__(self, num_features, out_dim, hid_dim, num_layers=5, layer_type='GCNConv'): 
        """
        Initialize a GNN model for graph classification. 
        args:
        """
        super(GNN, self).__init__()

        # choose the layer type
        Layer = globals()[layer_type]

        # construct layers
        self.conv1 = Layer(num_features, hid_dim)
        self.midde_layers = [
            Layer(hid_dim, hid_dim) for _ in range(num_layers-1)
        ]
        self.fc = Linear(hid_dim, out_dim)

    def forward(self, x, edge_list, batch):
        """
        Implement the GNN calculation. The output should be graph embeddings for the batch.

        args:
            x: a Tensor of shape [n, num_features], node features
            edge_list: a Tensor of shape [2, num_edges], each column is a tuple contains a pair `(sender, receiver)`. Here `sender` and `receiver`
            batch: the indicator vector indicating different graphs    
        """
        try:
            x = self.conv1(x, edge_list)
        except:
            print(f"Error in GNN.forward() with x.shape={x.shape} and edge_list.shape={edge_list.shape}")
            raise ValueError
        
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        for layer in self.midde_layers:
            x = layer(x, edge_list)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)

        # pooling
        x = global_mean_pool(x, batch)

        # linear classifier
        x = self.fc(x)
        out = F.log_softmax(x, dim=-1)

        return out
