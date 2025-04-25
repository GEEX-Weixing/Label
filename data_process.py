import os
import math
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import sys
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def load_dataset(dataset_name):  # 'cora', 'citeseer', 'pubmed'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("datas/ind.{}.{}".format(dataset_name, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append([pkl.load(f)])

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datas/ind.{}.test.index".format(dataset_name))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj, features
    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return adj_normalized
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)
    # return sparse_to_tuple(adj_normalized)

def normalize_adj_rw(adj):
    """Random walk transition matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    adj_normalized = d_mat_inv.dot(adj).tocoo()
    # return d_mat_inv.dot(adj).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)
    # return features

def get_input(dataset_name, device):
    adj, features = load_dataset(dataset_name)
    features = preprocess_features(features)
    n_nodes, feat_dim = features.shape
    # device = detect_device()
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    # adj_norm = preprocess_graph(adj)
    adj_norm = normalize_adj_rw(adj)
    symmetrical_adj = torch.tensor(normalize_adj(adj))
    adj_norm = torch.tensor(adj_norm)

    return features.to(device), adj_norm.to(device), symmetrical_adj.to(device)


def get_centers(indices, seed_labels, representation):
    centers = list()
    class_indices = list()
    seed_labels = seed_labels.detach().cpu().numpy()
    seeds_sorted = seed_labels.argsort()
    indices = indices[seeds_sorted]
    num0 = np.count_nonzero(seed_labels == 0)
    num1 = np.count_nonzero(seed_labels == 1)
    num2 = np.count_nonzero(seed_labels == 2)
    num3 = np.count_nonzero(seed_labels == 3)
    num4 = np.count_nonzero(seed_labels == 4)
    num5 = np.count_nonzero(seed_labels == 5)
    num6 = np.count_nonzero(seed_labels == 6)
    class0_indices = indices[: num0]
    class1_indices = indices[num0: num0+num1]
    class2_indices = indices[num0+num1: num0+num1+num2]
    class3_indices = indices[num0+num1+num2: num0+num1+num2+num3]
    class4_indices = indices[num0+num1+num2+num3: num0+num1+num2+num3+num4]
    class5_indices = indices[num0+num1+num2+num3+num4: num0+num1+num2+num3+num4+num5]
    class6_indices = indices[num0+num1+num2+num3+num4+num5: num0+num1+num2+num3+num4+num5+num6]
    class_indices.append(class0_indices)
    class_indices.append(class1_indices)
    class_indices.append(class2_indices)
    class_indices.append(class3_indices)
    class_indices.append(class4_indices)
    class_indices.append(class5_indices)
    class_indices.append(class6_indices)
    class0_center = representation[class0_indices].mean(axis=0)
    class1_center = representation[class1_indices].mean(axis=0)
    class2_center = representation[class2_indices].mean(axis=0)
    class3_center = representation[class3_indices].mean(axis=0)
    class4_center = representation[class4_indices].mean(axis=0)
    class5_center = representation[class5_indices].mean(axis=0)
    class6_center = representation[class6_indices].mean(axis=0)
    centers.append(class0_center)
    centers.append(class1_center)
    centers.append(class2_center)
    centers.append(class3_center)
    centers.append(class4_center)
    centers.append(class5_center)
    centers.append(class6_center)
    centers = np.array(centers)
    return centers, class_indices

def get_extended_vectors_indices(cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, class_indices):
    clusters = list()
    clusters.append(cluster0)
    clusters.append(cluster1)
    clusters.append(cluster2)
    clusters.append(cluster3)
    clusters.append(cluster4)
    clusters.append(cluster5)
    clusters.append(cluster6)
    intersections = list()
    for i in range(len(clusters) - 1):
        for j in range(i+1, len(clusters)):
            inter = set(clusters[i]) & set(clusters[j])
            intersections.append(list(inter))
    intersections_list = set([element for sublist in intersections for element in sublist])
    no_rep_cluster0 = [i for i in cluster0 if i not in intersections_list]
    no_rep_cluster1 = [i for i in cluster1 if i not in intersections_list]
    no_rep_cluster2 = [i for i in cluster2 if i not in intersections_list]
    no_rep_cluster3 = [i for i in cluster3 if i not in intersections_list]
    no_rep_cluster4 = [i for i in cluster4 if i not in intersections_list]
    no_rep_cluster5 = [i for i in cluster5 if i not in intersections_list]
    no_rep_cluster6 = [i for i in cluster6 if i not in intersections_list]
    extended_cluster0 = list(set(no_rep_cluster0) | set(class_indices[0]))
    extended_cluster1 = list(set(no_rep_cluster1) | set(class_indices[1]))
    extended_cluster2 = list(set(no_rep_cluster2) | set(class_indices[2]))
    extended_cluster3 = list(set(no_rep_cluster3) | set(class_indices[3]))
    extended_cluster4 = list(set(no_rep_cluster4) | set(class_indices[4]))
    extended_cluster5 = list(set(no_rep_cluster5) | set(class_indices[5]))
    extended_cluster6 = list(set(no_rep_cluster6) | set(class_indices[6]))
    return extended_cluster0, extended_cluster1, extended_cluster2, extended_cluster3, extended_cluster4, \
        extended_cluster5, extended_cluster6

def calculate_purity(gt_labels, cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6):
    gt_labels = gt_labels.detach().cpu().numpy()
    acc_num0 = 0
    acc_num1 = 0
    acc_num2 = 0
    acc_num3 = 0
    acc_num4 = 0
    acc_num5 = 0
    acc_num6 = 0
    for index in cluster0:
        if gt_labels[index] == 0:
            acc_num0 += 1
    for index in cluster1:
        if gt_labels[index] == 1:
            acc_num1 += 1
    for index in cluster2:
        if gt_labels[index] == 2:
            acc_num2 += 1
    for index in cluster3:
        if gt_labels[index] == 3:
            acc_num3 += 1
    for index in cluster4:
        if gt_labels[index] == 4:
            acc_num4 += 1
    for index in cluster5:
        if gt_labels[index] == 5:
            acc_num5 += 1
    for index in cluster6:
        if gt_labels[index] == 6:
            acc_num6 += 1
    purity = (acc_num0 + acc_num1 + acc_num2 + acc_num3 + acc_num4 + acc_num5 + acc_num6) / (len(cluster0) + len(cluster1)+
                len(cluster2) + len(cluster3) + len(cluster4) + len(cluster5) + len(cluster6) + 1)
    purity0 = round(acc_num0 / (len(cluster0)+1), 5)
    purity1 = round(acc_num1 / (len(cluster1)+1), 5)
    purity2 = round(acc_num2 / (len(cluster2)+1), 5)
    purity3 = round(acc_num3 / (len(cluster3)+1), 5)
    purity4 = round(acc_num4 / (len(cluster4)+1), 5)
    purity5 = round(acc_num5 / (len(cluster5)+1), 5)
    purity6 = round(acc_num6 / (len(cluster6)+1), 5)

    return purity, purity0, purity1, purity2, purity3, purity4, purity5, purity6


def Ncontrast(x_dis, adj, tau=1):
    """
    compute the Ncontrast loss
    """
    adj = adj.to_dense()
    adj_label = adj @ adj
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_distance_new(hidden, embedding):
    v1_norm = hidden / hidden.norm(dim=1, keepdim=True)
    v2_norm = embedding / embedding.norm(dim=1, keepdim=True)

    # Compute cosine similarity for each node
    cosine_similarities = (v1_norm * v2_norm).sum(dim=1)

    # Create a diagonal matrix with cosine similarities on the diagonal
    cosine_similarity_matrix = torch.diag(cosine_similarities)
    return cosine_similarity_matrix


def exp_diagonal_matrix(diagonal_matrix):
    """
    Applies the exponential function to the diagonal elements of a diagonal matrix.

    Parameters:
    diagonal_matrix (torch.Tensor): A square diagonal matrix of shape (N, N).

    Returns:
    torch.Tensor: A diagonal matrix with exponential values of the original diagonal elements.
    """
    # Ensure the input is a square matrix
    assert diagonal_matrix.shape[0] == diagonal_matrix.shape[1], "Input must be a square matrix."

    # Convert to dense matrix if it's sparse
    if diagonal_matrix.is_sparse:
        diagonal_matrix = diagonal_matrix.to_dense()

    # Extract the diagonal elements
    diagonal_elements = torch.diag(diagonal_matrix)

    # Apply the exponential function to the diagonal elements
    exp_diagonal_elements = torch.exp(diagonal_elements)

    # Create a new diagonal matrix with the exponential values
    exp_diagonal_matrix = torch.diag(exp_diagonal_elements)

    return exp_diagonal_matrix

def new_Ncontrast(x_dis, adj, x_dis_2, tau=1):
    adj = adj.to_dense()
    adj_label = adj
    x_dis = torch.exp(x_dis)
    x_dis_2 = exp_diagonal_matrix(x_dis_2)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    x_dis_sum_pos_2 = torch.sum(x_dis_2 * adj_label, 1)
    # x_dis_sum_pos_2 = torch.sum(x_dis_2 * adj_label, 1)
    loss = -torch.log((x_dis_sum_pos+x_dis_sum_pos_2) * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss

def Ncontrast1(x_dis, adj, tau=1):
    """
    compute the Ncontrast loss
    """
    adj = adj.to_dense()
    adj_label = adj
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def Ncontrast3(x_dis, adj, tau=1):
    """
    compute the Ncontrast loss
    """
    adj = adj.to_dense()
    adj_label = adj @ adj @ adj
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def rec_Ncontrastive4(x_dis, adj, tau=1):
    """
            compute the Ncontrast loss
            """
    adj = adj.to_dense()
    adj_label = adj @ adj @ adj @ adj
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss

def Ncontrast4(x_dis, adj, tau=1):
    """
    compute the Ncontrast loss
    """
    adj = adj.to_dense()
    adj_label = adj @ adj @ adj @ adj
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def Concat_Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def Ncontrast5(x_dis, adj, tau=1):
    """
    compute the Ncontrast loss
    """
    adj = adj.to_dense()
    adj_label = adj @ adj @ adj @ adj @ adj
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    # mask = torch.eye(x_dis.shape[0]).cuda()
    mask = torch.eye(x_dis.shape[0])
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def self_correct(p_max, class0_indices, class1_indices, class2_indices, class3_indices, class4_indices, class5_indices, class6_indices, logits):
    new_class0_indices = list()
    new_class1_indices = list()
    new_class2_indices = list()
    new_class3_indices = list()
    new_class4_indices = list()
    new_class5_indices = list()
    new_class6_indices = list()
    for i in class0_indices:
        if logits[i][0] >= p_max:
            new_class0_indices.append(i)
    for i in class1_indices:
        if logits[i][1] >= p_max:
            new_class1_indices.append(i)
    for i in class2_indices:
        if logits[i][2] >= p_max:
            new_class2_indices.append(i)
    for i in class3_indices:
        if logits[i][3] >= p_max:
            new_class3_indices.append(i)
    for i in class4_indices:
        if logits[i][4] >= p_max:
            new_class4_indices.append(i)
    for i in class5_indices:
        if logits[i][5] >= p_max:
            new_class5_indices.append(i)
    for i in class6_indices:
        if logits[i][6] >= p_max:
            new_class6_indices.append(i)
    return new_class0_indices, new_class1_indices, new_class2_indices, new_class3_indices, new_class4_indices, new_class5_indices, new_class6_indices

def N2N_MIM_Loss_2(embeddings, anchor_indices, anchor_neighbor, other0, other1, other2, other3, other4, other5):
    center_embedding = torch.mean(embeddings[anchor_indices], dim=0)
    total_positive_embeddings = embeddings[anchor_neighbor]
    others = other0 + other1 + other2 + other3 + other4 + other5

    tau = 0.5

    rows_to_exclude = anchor_neighbor
    # rows_to_select = [i for i in range(len(embeddings)) if i not in rows_to_exclude]
    total_negative_embeddings = embeddings[others]

    center_embedding = F.normalize(center_embedding, p=2, dim=0)
    total_positive_embeddings = F.normalize(total_positive_embeddings, p=2, dim=1)
    total_negative_embeddings = F.normalize(total_negative_embeddings, p=2, dim=1)

    positive_score = torch.exp((center_embedding @ total_positive_embeddings.T) / tau).sum(-1)
    negative_score = torch.exp((center_embedding @ total_negative_embeddings.T) / tau).sum(-1)
    con_loss = -torch.log(positive_score + 1e-8 / (negative_score + 1e-8))
    return con_loss

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()



def calculate_class_centers_and_indices(logits_all, train_indices, train_labels, num_classes=40):
    """
    计算每个类别的 logits 中心，并返回每个类别对应的训练集节点索引。

    参数:
    - logits_all: numpy array，整体数据集的 logits 输出，大小为 [num_total_nodes, num_classes]。
    - train_indices: numpy array，训练集节点的索引，大小为 [num_train_nodes]。
    - train_labels: numpy array，训练集节点的标签，大小为 [num_train_nodes]。
    - num_classes: int，类别数，默认为 40。

    返回:
    - logits_centers: numpy array，大小为 [num_classes, num_features]，每个类别的 logits 中心。
    - class_indices: List[List[int]]，二维列表，每个子列表包含属于该类别的节点索引。
    """
    # 提取训练集节点的 logits 输出
    train_logits = logits_all[train_indices]
    # print(train_logits)

    # 初始化一个二维列表来存储每个类别的节点索引
    class_indices = [[] for _ in range(num_classes)]  # 40个类别的索引列表

    # 初始化一个 numpy array 来存储每个类别的 logits 中心，大小为 [num_classes, num_features]
    logits_centers = np.zeros((num_classes, logits_all.shape[1]))  # 每个类别的 logits 中心

    # 计算每个类别的 logits 中心，并填充 class_indices
    for i in range(num_classes):
        # 获取属于当前类别的训练集节点的索引
        class_mask = (train_labels == i)
        class_indices[i] = train_indices[class_mask].tolist()  # 获取该类别的节点索引

        # 计算该类别的 logits 中心
        logits_centers[i] = train_logits[class_mask].mean(axis=0)

    return logits_centers, class_indices


def get_flatten(new_clusters_indices, num_classes):
    lists = []
    for i in range(num_classes):
        class_neighbor_indices = list(np.array(new_clusters_indices[i]).ravel())
        lists.append(class_neighbor_indices)
    return lists


from collections import OrderedDict

def merge_and_remove_duplicates(expansion_before_indices, expansion_after_indices):
    """
    合并两个列表，并确保每个类别的节点索引不重复。

    参数:
    - expansion_before_indices: List[List[int]]，扩充前的类别节点索引列表。
    - expansion_after_indices: List[List[int]]，扩充后的类别节点索引列表。

    返回:
    - merged_indices: List[List[int]]，合并并去重后的类别节点索引列表。
    """
    merged_indices = []

    for before_indices, after_indices in zip(expansion_before_indices, expansion_after_indices):
        # 合并扩充前和扩充后的节点索引
        # print('++++++++++++++++++++++++++++++++')
        # print(before_indices, len(before_indices))
        # print(after_indices, len(after_indices))
        combined_indices = before_indices + after_indices

        # 使用 OrderedDict 去重并保持顺序
        unique_indices = list(OrderedDict.fromkeys(combined_indices))

        # 将去重后的结果加入新的列表
        merged_indices.append(unique_indices)

    return merged_indices

def filter_nodes_by_confidence(logits, class_indices, threshold):
    filtered_class_indices = []

    # 遍历每个类别
    for indices in class_indices:
        # 获取当前类别对应的节点的logits
        class_logits = logits[indices]  # 形状为 [num_class_nodes, num_classes]

        # 计算每个节点的最大置信度（每行的最大值）
        max_confidences = np.max(class_logits, axis=1)

        # 筛选出置信度大于阈值的节点索引
        filtered_indices = [indices[i] for i in range(len(indices)) if max_confidences[i] > threshold]

        # 将筛选后的节点索引加入结果列表
        filtered_class_indices.append(filtered_indices)

    return filtered_class_indices

def calculate_loss(expansion_before_indices, expansion_after_indices, embeddings, num_classes, temperature=0.5):
    """
    对比学习损失函数：扩充前节点与中心拉近，扩充后节点与其他类别中心拉远。

    参数:
    - expansion_before_indices: List[List[int]]，扩充前的类别节点索引。
    - expansion_after_indices: List[List[int]]，扩充后的类别节点索引。
    - embeddings: Tensor，所有节点的嵌入表示，大小为 [num_nodes, embedding_dim]，支持梯度计算。
    - num_classes: int，类别数。
    - temperature: float，对比学习的温度参数，控制相似度的分布。

    返回:
    - loss: Tensor，计算得到的总损失值。
    """
    loss = 0.0

    # 计算每个类别的嵌入中心（扩充前的节点中心）
    class_centers = []
    for i in range(num_classes):
        before_indices = expansion_before_indices[i]
        class_center = embeddings[before_indices].mean(dim=0)  # 对扩充前节点求均值作为中心
        class_centers.append(class_center)

    # 将类别中心转化为 tensor
    class_centers = torch.stack(class_centers)

    # 正样本对：扩充前节点与其类别中心
    for i in range(num_classes):
        # 获取扩充前和扩充后的节点
        before_indices = expansion_before_indices[i]
        after_indices = expansion_after_indices[i]

        # 获取扩充前节点和该类别中心的嵌入
        before_embeddings = embeddings[before_indices]
        center = class_centers[i]

        # 计算扩充前节点与该类别中心的余弦相似度
        cos_sim_before = F.cosine_similarity(before_embeddings,
                                             center.unsqueeze(0).repeat(before_embeddings.size(0), 1), dim=1)

        # 正样本损失：最小化节点与中心的距离（最大化相似度）
        loss += torch.sum(1 - cos_sim_before)  # 拉近：节点与中心的相似度最大化（最小化距离）

        # 拉远：扩充后节点与其他类别中心
        after_embeddings = embeddings[after_indices]

        for j in range(num_classes):
            if i != j:  # 计算扩充后的节点与其他类别中心的相似度
                other_center = class_centers[j]
                cos_sim_after = F.cosine_similarity(after_embeddings,
                                                    other_center.unsqueeze(0).repeat(after_embeddings.size(0), 1),
                                                    dim=1)

                # 负样本损失：最小化节点与其他类别中心的相似度（最小化相似度，拉远）
                loss += torch.sum(torch.exp(cos_sim_after / temperature))  # 拉远：节点与其他类别中心的距离最大化（最小化相似度）

    return loss / num_classes


def ca(out, labels, evaluator, val_idx, test_idx):
    val_acc = evaluator.eval({
        'y_true': labels[val_idx],
        'y_pred': out.argmax(dim=-1, keepdim=True)[val_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[test_idx],
        'y_pred': out.argmax(dim=-1, keepdim=True)[test_idx],
    })['acc']
    return val_acc, test_acc

def update_labels_and_mask(mask, labels, expanded_indices, all_predictions, num_classes):
    updated_labels = labels.clone()  # 复制原始的 labels
    updated_train_mask = mask.clone()  # 复制原始的 train_mask

    # 扩充后节点的标签更新
    for i in range(num_classes):
        # 获取扩充后该类别的节点索引
        expanded_nodes = expanded_indices[i]

        # 更新扩充后节点的标签
        with torch.no_grad():
            # 根据模型对所有节点的预测，获取该类别的预测标签
            predicted_labels = all_predictions[expanded_nodes].argmax(dim=1)  # 获取每个节点的最大预测类别

            # 将扩充后节点的标签更新为模型的预测标签
            updated_labels[expanded_nodes] = predicted_labels

        # 更新扩充后节点的 train_mask
        updated_train_mask[expanded_nodes] = True  # 扩充后节点变为训练集的一部分

    return updated_labels, updated_train_mask

from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

def edge_index2adj_t(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    # 将edge_index转换为SparseTensor格式的adj_t
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    return adj_t

