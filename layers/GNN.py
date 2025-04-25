import torch
# from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, APPNP
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs

class GCN3(nn.Module):
	def __init__(self, input_dim, hidden_dim, out_dim):
		super(GCN3, self).__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, out_dim)
		self.softmax = nn.Softmax()

	def forward(self, x, edge_index, edge_weight=None):
		x = self.conv1(x, edge_index, edge_weight=edge_weight)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index, edge_weight=edge_weight)
		logits = self.softmax(x)
		preds = torch.argmax(logits, dim=1)
		return logits, preds, x

class GCN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
		super(GCN, self).__init__()
		self.convs = torch.nn.ModuleList()
		self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
		self.bns = torch.nn.ModuleList()
		self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
		for _ in range(num_layers - 2):
			self.convs.append(
				GCNConv(hidden_channels, hidden_channels, cached=True))
			self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
		self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

		self.dropout = dropout
		self.softmax = nn.Softmax()

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		for bn in self.bns:
			bn.reset_parameters()

	def forward(self, x, adj_t, adj_t2):
		if adj_t != None:
			if adj_t2 != None:
				for i, conv in enumerate(self.convs[:-1]):
					x = conv(x, adj_t)
					x = self.bns[i](x)
					x = F.relu(x)
					embeddings = F.dropout(x, p=self.dropout, training=self.training)
				x = self.convs[-1](embeddings, adj_t)
				x_2 = self.convs[-1](embeddings, adj_t2)
				logits = self.softmax(x)
				logits_2 = self.softmax(x_2)
				predictions = torch.argmax(logits, dim=1)
			else:
				for i, conv in enumerate(self.convs[:-1]):
					x = conv(x, adj_t)
					x = self.bns[i](x)
					x = F.relu(x)
					embeddings = F.dropout(x, p=self.dropout, training=self.training)
				x = self.convs[-1](embeddings, adj_t)
				logits = self.softmax(x)
				logits_2 = None
				predictions = torch.argmax(logits, dim=1)
		else:
			for i, conv in enumerate(self.convs[:-1]):
				x = conv(x, adj_t2)
				x = self.bns[i](x)
				x = F.relu(x)
				embeddings = F.dropout(x, p=self.dropout, training=self.training)
			# x = self.convs[-1](x, adj_t2)
			x_2 = self.convs[-1](embeddings, adj_t2)
			logits = None
			logits_2 = self.softmax(x_2)
			predictions = torch.argmax(logits_2, dim=1)
		return logits, predictions, logits_2, embeddings



import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.softmax = nn.Softmax()
        self.softmax2 = nn.Softmax()

    def forward(self, x, edge_index, edge_index_2):
        x = F.dropout(x, p=0.6, training=self.training)
        embeddings = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(embeddings, p=0.6, training=self.training)
        x_1 = self.conv2(embeddings, edge_index)
        x_2 = self.conv3(embeddings, edge_index_2)

        logits = self.softmax(x_1)
        logits_2 = self.softmax2(x_2)
        predictions = torch.argmax(logits, dim=1)
        return logits, logits_2, predictions

import torch.nn as nn
import torch
import torch.nn.functional as F

class SGC(nn.Module):
    def __init__(self, k, nfeat, n_hidden, n_class):
        super(SGC, self).__init__()
        self.k = k
        self.W = nn.Linear(nfeat, n_hidden)
        self.M = nn.Linear(n_hidden, n_class)
        self.softmax = nn.Softmax()


    def forward(self, x, adj):
        k = self.k
        for i in range(k):
            x = torch.spmm(adj, x)
        # embeddings = x
        embeddings = self.W(x)
        embedding = F.relu(embeddings)
        logits = self.M(embedding)
        logits = self.softmax(logits)
        predictions = torch.argmax(logits, dim=1)
        return logits, predictions, embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.softmax = nn.Softmax()
        self.softmax_2 = nn.Softmax()
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.conv3 = SAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_index_2):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        embeddings = F.relu(x)
        # x = F.dropout(embeddings, training=self.training)
        x_1 = self.conv2(embeddings, edge_index)
        x_2 = self.conv3(embeddings, edge_index_2)
        logits = self.softmax(x_1)
        logits_2 = self.softmax(x_2)
        predictions = torch.argmax(logits, dim=1)

        return logits, logits_2, predictions

import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv

class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, K=2)  # K是卷积层数
        self.conv_2 = SGConv(in_channels, out_channels, K=2)  # K是卷积层数
        self.softmax = nn.Softmax()
        self.softmax_2 = nn.Softmax()

    def forward(self, x, edge_index, edge_index_2):
        # x, edge_index = data.x, data.edge_index
        x_1 = self.conv(x, edge_index)
        x_2 = self.conv_2(x, edge_index_2)
        logits = self.softmax(x_1)
        logits_2 = self.softmax_2(x_2)

        predictions = torch.argmax(logits, dim=1)
        return logits, logits_2, predictions


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_sparse
from torch import FloatTensor
import numpy as np


class H2GCN(nn.Module):
	def __init__(
			self,
			feat_dim: int,
			hidden_dim: int,
			class_dim: int,
			k: int = 2,
			dropout: float = 0.5,
			use_relu: bool = True
	):
		super(H2GCN, self).__init__()
		self.dropout = dropout
		self.k = k
		self.act = F.relu if use_relu else lambda x: x
		self.use_relu = use_relu
		# self.w_embed = nn.Parameter(torch.zeros(size=(feat_dim, hidden_dim)), requires_grad=True).cuda()
		self.w_embed = nn.Parameter(torch.zeros(size=(feat_dim, hidden_dim)), requires_grad=True)
		self.w_embed_2 = nn.Parameter(torch.zeros(size=(feat_dim, hidden_dim)), requires_grad=True)

		self.w_classify = nn.Parameter(
			torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
			requires_grad=True
		)
		self.w_classify_2 = nn.Parameter(
			torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
			requires_grad=True
		)
		self.params = [self.w_embed, self.w_classify]
		self.params_2 = [self.w_embed_2, self.w_classify_2]
		self.initialized = False
		self.a1 = None
		self.a2 = None
		self.reset_parameter()

	def reset_parameter(self):
		nn.init.xavier_uniform_(self.w_embed)
		nn.init.xavier_uniform_(self.w_classify)
		nn.init.xavier_uniform_(self.w_embed_2)
		nn.init.xavier_uniform_(self.w_classify_2)

	@staticmethod
	def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
		csp = sp_tensor.coalesce()
		return torch.sparse_coo_tensor(
			indices=csp.indices(),
			values=torch.where(csp.values() > 0, 1, 0),
			size=csp.size(),
			dtype=torch.float
		)

	@staticmethod
	def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
		assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
		sp1, sp2 = sp1.coalesce(), sp2.coalesce()
		index1, value1 = sp1.indices(), sp1.values()
		index2, value2 = sp2.indices(), sp2.values()
		m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
		indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
		return torch.sparse_coo_tensor(
			indices=indices,
			values=values,
			size=(m, k),
			dtype=torch.float
		)

	@classmethod
	def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
		n = adj.size(0)
		d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
		d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
		d_tiled = torch.sparse_coo_tensor(
			indices=[list(range(n)), list(range(n))],
			values=d_diag,
			size=(n, n)
		)
		return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

	def _prepare_prop(self, adj, adj_2):
		n = adj.size(0)
		device = adj.device
		self.initialized = True
		sp_eye = torch.sparse_coo_tensor(
			indices=[list(range(n)), list(range(n))],
			values=[1.0] * n,
			size=(n, n),
			dtype=torch.float
		).to(device)
		# initialize A1, A2
		a1 = self._indicator(adj - sp_eye)
		a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
		a3 = self._indicator(adj_2 - sp_eye)
		a4 = self._indicator(self._spspmm(adj_2, adj_2) - adj_2 - sp_eye)

		self.a1 = self._adj_norm(a1)
		self.a2 = self._adj_norm(a2)
		self.a3 = self._adj_norm(a3)
		self.a4 = self._adj_norm(a4)


	def forward(self, adj, adj_2, x):
		if not self.initialized:
			self._prepare_prop(adj, adj_2)

		rs = [self.act(torch.mm(x, self.w_embed))]
		rs_2 = [self.act(torch.mm(x, self.w_embed_2))]
		for i in range(self.k):
			r_last = rs[-1]
			r1 = torch.spmm(self.a1, r_last)
			r2 = torch.spmm(self.a2, r_last)
			rs.append(self.act(torch.cat([r1, r2], dim=1)))
			r_last_2 = rs_2[-1]
			r3 = torch.spmm(self.a3, r_last_2)
			r4 = torch.spmm(self.a4, r_last_2)
			rs_2.append(self.act(torch.cat([r3, r4], dim=1)))
		r_final = torch.cat(rs, dim=1)
		r_final_2 = torch.cat(rs_2, dim=1)
		r_final = F.dropout(r_final, self.dropout, training=self.training)
		r_final_2 = F.dropout(r_final_2, self.dropout, training=self.training)
		logits = torch.softmax(torch.mm(r_final, self.w_classify), dim=1)
		logits_2 = torch.softmax(torch.mm(r_final_2, self.w_classify_2), dim=1)
		prediction = torch.argmax(logits, dim=1)
		return logits, logits_2, prediction

