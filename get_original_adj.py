import numpy as np

from data_process import *
from src.data_hete import Dataset
from torch_geometric.datasets import Planetoid
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
device = f'cuda:{3}' if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
data = Dataset(root="data", dataset='cora')[0].to(device)
# print(data)
# edge_index = data.edge_index
edge_index_numpy = data.edge_index.detach().cpu().numpy()

# Get the number of nodes in the graph
num_nodes = data.num_nodes

# Create an empty adjacency matrix
adjacency_matrix = np.zeros((num_nodes, num_nodes))

# Populate the adjacency matrix based on edge_index
adjacency_matrix[edge_index_numpy[0], edge_index_numpy[1]] = 1
adjacency_matrix[edge_index_numpy[1], edge_index_numpy[0]] = 1
adjacency_matrix = adjacency_matrix + np.array(torch.eye(adjacency_matrix.shape[0]))
np.save('original_adj.npy', adjacency_matrix)



