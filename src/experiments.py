from torch_geometric.loader import DataLoader
from train import train
import numpy as np
import pickle


# You are supposed to call your train function to learn a model here. 
model = train(trainset)

# Below your model is called to predict test labels 
model.eval()

with open("dataset/test.pickle", "rb") as input_file:
    testset = pickle.load(input_file)

test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
for batch in test_loader:
    pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)

# Save predictions to the .txt file
save_file = "gnn_predictions.txt"
print("%d model predictions saved to %s" % (pred.shape[0], save_file))
np.savetxt(save_file, pred)