from torch_geometric.data import InMemoryDataset, Dataset
from load_call_graph import call_graph_dataset

# for constructing the call graph dataset
class CallGraphDataset(InMemoryDataset):
    def __init__(self, file_num=1, graph_num=1000, root=None, transform=None, pre_transform=None):
        super(CallGraphDataset, self).__init__(root, transform, pre_transform)
        data_list = call_graph_dataset(file_num, graph_num)
        # print(f"Loaded {len(data_list)} graphs with type {type(data_list[0])}")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass

class CallGraphPairDataset(Dataset):
    def __init__(self, graph_data, similarity_scores):
        self.graph_data = graph_data
        self.similarity_scores = similarity_scores

    def __len__(self):
        return len(self.graph_data) // 2

    def __getitem__(self, index):
        graph1 = self.graph_data[index * 2]
        graph2 = self.graph_data[index * 2 + 1]
        similarity_score = self.similarity_scores[index]
        return graph1, graph2, similarity_score

# Example usage
# dataset = GraphPairDataset(graph_data, similarity_scores)
# dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# for batch in dataloader:
#     inputs, similarity_scores = batch
#     # Do something with the inputs and similarity scores
