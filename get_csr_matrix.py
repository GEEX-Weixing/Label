# from layers.gae import GAE, detect_device
import torch
import os
import copy
import torch
from src.data_arxiv import Dataset
from data_process import *
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz

def detect_device():
    if torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    return device


device = detect_device()
data = Dataset(root="data", dataset='ogbn-arxiv')[0].to(device)
edge_index = data.edge_index

# 将edge_index转换为COO格式
edge_index_coo = coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                            shape=(data.num_nodes, data.num_nodes))

# 将COO格式的稀疏矩阵转换为CSR格式
adj_matrix_csr = edge_index_coo.tocsr()
# print(adj_matrix_csr)
save_npz('adj_csr_matrix.npz', adj_matrix_csr)

features = data.x
dense_array = features.detach().cpu().numpy()
from scipy.sparse import lil_matrix
lil_matrixs = lil_matrix(dense_array)
csr_matrix = lil_matrixs.tocsr()
save_npz('features_csr_matrix.npz', csr_matrix)
