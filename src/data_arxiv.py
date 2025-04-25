import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
import json
from torch import Tensor, LongTensor
from torch_geometric.utils import to_undirected, is_undirected
from typing import (
    Union,
    Optional,
)
from itertools import chain
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def safe_to_undirected(
        edge_index: LongTensor,
        edge_attr: Optional[Tensor] = None):
    if is_undirected(edge_index, edge_attr):
        return edge_index, edge_attr
    else:
        return to_undirected(edge_index, edge_attr)


def get_data_arxiv(dataset_name='arxiv'):
    # with open('wiki-cs/dataset/data.json', 'r', encoding='utf-8') as f:
    #     datas = json.load(f)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())
    # da_p = Dataset(root='data', dataset='pubmed')[0]
    data = dataset[0]
    adj_t = data.adj_t  # SparseTensor格式的邻接矩阵
    row, col, _ = adj_t.coo()

    # 将 row 和 col 组合成 edge_index
    data.edge_index = torch.stack([row, col], dim=0)

    nodes = torch.tensor(np.arange(data.x.size(0)), dtype=torch.long)

    create_masks(data=data, name=dataset_name)

    data = Data(nodes=nodes, adj_t=data.adj_t, edge_index=data.edge_index, edge_attr=data.edge_attr, x=data.x, y=data.y.squeeze(),
                train_mask0_1=data.train_mask0_1, val_mask0_1=data.val_mask0_1, test_mask0_1=data.test_mask0_1,
                train_mask0_2=data.train_mask0_2, val_mask0_2=data.val_mask0_2, test_mask0_2=data.test_mask0_2,
                train_mask0_3=data.train_mask0_3, val_mask0_3=data.val_mask0_3, test_mask0_3=data.test_mask0_3,
                num_nodes=data.num_nodes)

    return data

def create_masks(data, name='cora'):

    labels = data.y.squeeze()
    mask0_1 = Random_Nonzero_Masking(data, labels, label_rate=0.1)
    mask0_2 = Random_Nonzero_Masking(data, labels, label_rate=0.2)
    mask0_3 = Random_Nonzero_Masking(data, labels, label_rate=0.3)

    data.train_mask0_1, data.val_mask0_1, data.test_mask0_1 = mask0_1
    data.train_mask0_2, data.val_mask0_2, data.test_mask0_2 = mask0_2
    data.train_mask0_3, data.val_mask0_3, data.test_mask0_3 = mask0_3

    return data

def Random_Nonzero_Masking(data, labels, label_rate):
    label_rate *= 0.01
    train_size = int(data.x.size(0) * label_rate)
    eval_size = data.x.size(0) - train_size

    dev_size = int(eval_size * 0.1)

    final_train_mask = None;
    final_val_mask = None;
    final_test_mask = None
    cnt = 0
    while True:
        labels = data.y.numpy()

        perm = np.random.permutation(labels.shape[0])
        train_index = perm[:train_size]
        dev_index = perm[train_size: train_size + dev_size]
        test_index = perm[train_size + dev_size:]

        data_index = np.arange(labels.shape[0])
        train_mask = torch.tensor(np.in1d(data_index, train_index), dtype=torch.bool)
        dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
        test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)

        train_mask = train_mask.reshape(1, -1)
        test_mask = test_mask.reshape(1, -1)
        dev_mask = dev_mask.reshape(1, -1)

        if np.unique(labels).shape[0] == np.unique(labels[train_mask[0]]).shape[0]:
            cnt += 1
        else:
            continue

        if final_train_mask is None:
            final_train_mask = train_mask
            final_val_mask = dev_mask
            final_test_mask = test_mask
        else:
            final_train_mask = torch.cat((final_train_mask, train_mask), dim=0)
            final_val_mask = torch.cat((final_val_mask, dev_mask), dim=0)
            final_test_mask = torch.cat((final_test_mask, test_mask), dim=0)

        if cnt == 20:
            break

    return final_train_mask, final_val_mask, final_test_mask




