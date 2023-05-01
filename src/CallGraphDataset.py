from torch_geometric.data import InMemoryDataset
from load_call_graph import call_graph_dataset

# for constructing the call graph dataset
class CallGraphDataset(InMemoryDataset):
    def __init__(self, file_num=1, root=None, transform=None, pre_transform=None):
        super(CallGraphDataset, self).__init__(root, transform, pre_transform)
        data_list = call_graph_dataset(file_num)
        # print(f"Loaded {len(data_list)} graphs with type {type(data_list[0])}")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass
