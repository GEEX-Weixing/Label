from torch.optim import Adam
from copy import deepcopy
from embedder import embedder
from torch_geometric.utils import to_dense_adj
from data_process import *
import faiss
from src.utils import *
from layers.GNN import GCN, GCN3
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import to_scipy_sparse_matrix


class CL_Trainer_M(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def _init_model(self):
        self.model = GCN(self.data.x.size(1), 256, self.num_classes).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        self.optimizer_n2c = Adam(self.model.parameters(), lr=self.args.lr * 2, weight_decay=self.args.decay)

    def _init_dataset(self):

        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)
        self.running_train_idx = torch.nonzero(self.running_train_mask).squeeze().to(self.device)
        self.running_val_mask = deepcopy(self.val_mask)
        self.running_val_idx = torch.nonzero(self.running_val_mask).squeeze().to(self.device)
        self.running_test_mask = deepcopy(self.test_mask)
        self.running_test_idx = torch.nonzero(self.running_test_mask).squeeze().to(self.device)
        self.running_edge_index = deepcopy(self.edge_index)
        self.running_train_loader_1 = NeighborSampler(self.running_edge_index, node_idx=self.running_train_idx,
                                                      sizes=[-1], batch_size=1024, shuffle=True, num_workers=12)
        self.running_train_loader_2 = NeighborSampler(self.edge_index_2, node_idx=self.running_train_idx, sizes=[-1],
                                                      batch_size=1024, shuffle=True, num_workers=12)
        eta = self.data.num_nodes / (to_dense_adj(self.data.edge_index).sum() / self.data.num_nodes) ** len(
            self.hidden_layers)
        self.t = (self.labels[self.train_mask].unique(return_counts=True)[1] * 3 * eta / len(
            self.labels[self.train_mask])).type(torch.int64)
        self.t = self.t / self.args.stage

    def pretrain(self, mask, stage):
        reset_parameters(self.model)
        for epoch in range(400):
            if self.args.dataset != 'arxiv':
                self.model.train()
                self.optimizer.zero_grad()
                logits, preds, logits_2, embeddings = self.model(self.features,
                                                                 edge_index2adj_t(self.running_edge_index,
                                                                                  self.features.size(0)),
                                                                 edge_index2adj_t(self.edge_index_2,
                                                                                  self.features.size(0)))
                ce_loss_1 = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                ce_loss_2 = F.cross_entropy(logits_2[self.running_train_mask], self.labels[self.running_train_mask])
                x_dis_1 = get_feature_dis(logits)
                x_dis_1_2 = get_distance_new(logits, logits_2)
                nc_loss_1 = new_Ncontrast(x_dis_1, to_dense_adj(self.running_edge_index), x_dis_1_2)

                x_dis_2 = get_feature_dis(logits_2)
                x_dis_2_2 = get_distance_new(logits_2, logits)
                nc_loss_2 = new_Ncontrast(x_dis_2, to_dense_adj(self.edge_index_2), x_dis_2_2)
                total_loss = 0.5 * (ce_loss_1 + ce_loss_2) + 0.05 * (nc_loss_1 + nc_loss_2)

                total_loss.backward()
                self.optimizer.step()
            else:
                self.model.train()
                self.optimizer.zero_grad()
                outs, preds, _, _ = self.model(self.features,
                                               edge_index2adj_t(self.running_edge_index, self.features.size(0)),
                                               edge_index2adj_t(self.edge_index_2, self.features.size(0)))
                # print(out.shape)
                loss = F.cross_entropy(outs[self.running_train_mask], self.labels[self.running_train_mask])
                loss.backward()
                self.optimizer.step()

                for batch_size, n_id, adjs in self.running_train_loader_1:
                    self.optimizer.zero_grad()
                    out, preds, _, _ = self.model(self.features[n_id], adjs[0].to(self.device), None)
                    x_dis = get_feature_dis(out, self.device)
                    row, col = adjs[0]
                    adj = torch.zeros((n_id.size(0), n_id.size(0)), device=self.device)
                    adj[row, col] = 1
                    nc_loss = Ncontrast1(x_dis, adj)
                    nc_loss.backward()
                    self.optimizer.step()
                for batch_size, n_id, adjs in self.running_train_loader_2:
                    self.optimizer.zero_grad()
                    _, preds, out, _ = self.model(self.features[n_id], None, adjs[0].to(self.device))
                    x_dis = get_feature_dis(out, self.device)
                    row, col = adjs[0]
                    adj = torch.zeros((n_id.size(0), n_id.size(0)), device=self.device)
                    adj[row, col] = 1
                    nc_loss_2 = Ncontrast1(x_dis, adj)
                    nc_loss_2.backward()
                    self.optimizer.step()
            if self.args.dataset != 'arxiv':
                train_acc, val_acc, test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask,
                                                                self.train_mask, self.val_mask,
                                                                self.test_mask, self.device)
            else:
                _, test_acc = ca(out, self.data.y, self.evaluator, self.running_val_idx, self.running_test_idx)

            st = '[Fold: {}/{}][Stage : {}/{}][Epoch {}/{}] Test Accuracy: {:.4f}'.format(mask + 1, self.args.folds,
                                                                                          stage + 1, self.args.stage,
                                                                                          epoch + 1, 400,
                                                                                          test_acc.item())

            print(st)

    def adj_adjust(self, mask, stage):
        self.model.eval()
        logits, preds, _, embeddings = self.model(self.features,
                                                  edge_index2adj_t(self.running_edge_index, self.features.size(0)),
                                                  edge_index2adj_t(self.edge_index_2, self.features.size(0)))
        detect_logits = logits.detach().cpu().numpy()
        init_centers, class_indices = calculate_class_centers_and_indices(detect_logits,
                                                                          self.running_train_idx.detach().cpu().numpy(),
                                                                          self.labels[
                                                                              self.running_train_idx].detach().cpu().numpy())

        logits_eval = logits.detach().cpu().numpy()
        new_clusters_indices = list()
        xb = logits_eval.astype('float32')
        for i in range(self.num_classes):
            cluster_indices = class_indices[i]
            new_cluster_indices = list()
            for j in range(len(cluster_indices)):
                xq = np.expand_dims(logits_eval[cluster_indices[j]], axis=0).astype('float32')
                index = faiss.IndexFlatL2(self.num_classes)
                index.add(xb)
                k = self.args.edge_range
                class_cluster_indices = np.squeeze(index.search(xq, k)[1], axis=0)[1: k]
                new_cluster_indices.append(class_cluster_indices)
            new_clusters_indices.append(new_cluster_indices)

        class_neighbor_indices_list = get_flatten(new_clusters_indices, self.num_classes)

        extended_clusters = merge_and_remove_duplicates(class_indices, class_neighbor_indices_list)
        extended_clusters = filter_nodes_by_confidence(detect_logits, extended_clusters, self.p_max)
        print(extended_clusters)
        self.running_edge_index = self.update_edges_based_on_train_sets_mask(torch.tensor(logits_eval).to(self.device),
                                                                             class_indices, extended_clusters, stage,
                                                                             mask)

    def self_train(self, mask, stage):
        # 寻找各自的最近邻（K=20）
        self.model.eval()
        logits, preds, _, embeddings = self.model(self.features,
                                                  edge_index2adj_t(self.running_edge_index, self.features.size(0)),
                                                  edge_index2adj_t(self.edge_index_2, self.features.size(0)))
        logits = logits.detach().cpu().numpy()

        init_centers, class_indices = calculate_class_centers_and_indices(logits,
                                                                          self.running_train_idx.detach().cpu().numpy(),
                                                                          self.labels[
                                                                              self.running_train_idx].detach().cpu().numpy())

        new_clusters_indices = list()

        xb = logits.astype('float32')
        for i in range(self.num_classes):
            cluster_indices = class_indices[i]
            new_cluster_indices = list()
            for j in range(len(cluster_indices)):
                xq = np.expand_dims(logits[cluster_indices[j]], axis=0).astype('float32')
                index = faiss.IndexFlatL2(self.num_classes)
                index.add(xb)
                k = 21
                class_cluster_indices = np.squeeze(index.search(xq, k)[1], axis=0)[1: k]
                new_cluster_indices.append(class_cluster_indices)
            new_clusters_indices.append(new_cluster_indices)

        class_neighbor_indices_list = get_flatten(new_clusters_indices, self.num_classes)
        class_neighbor_indices_list = filter_nodes_by_confidence(logits, class_neighbor_indices_list, self.p_max)

        # 重置网络参数
        reset_parameters(self.model)
        for epoch in range(40):
            self.model.train()
            self.optimizer_n2c.zero_grad()
            logits, preds, logits_2, x = self.model(self.features,
                                                    edge_index2adj_t(self.running_edge_index, self.features.size(0)),
                                                    edge_index2adj_t(self.edge_index_2, self.features.size(0)))

            ce_loss = F.cross_entropy(logits[self.running_train_idx], self.labels[self.running_train_idx])
            contrastive_loss = calculate_loss(class_indices, class_neighbor_indices_list, logits, self.num_classes)
            # N2N Loss 提前做好的采样

            total_loss = ce_loss + contrastive_loss
            # total_loss.requires_grad_(True)

            total_loss.backward(retain_graph=True)
            self.optimizer_n2c.step()
            if self.args.dataset != 'arxiv':
                train_acc, val_acc, test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask,
                                                                self.train_mask, self.val_mask,
                                                                self.test_mask, self.device)
            else:
                _, test_acc = ca(logits, self.data.y, self.evaluator, self.running_val_idx, self.running_test_idx)

            st = '[Fold: {}/{}][Stage : {}/{}][Epoch {}/{}] CE Loss: {:.4f} N2C Loss: {:.4f} Loss: {:.4f} Test Accuracy: ' \
                 '{:.4f}'.format(mask + 1, self.args.folds, stage + 1, self.args.stage, epoch + 1, 400,
                                 ce_loss.item(), contrastive_loss.item(), total_loss.item(), test_acc)

            print(st)

        self.model.eval()
        logits, preds, _, _ = self.model(self.features,
                                         edge_index2adj_t(self.running_edge_index, self.features.size(0)),
                                         edge_index2adj_t(self.edge_index_2, self.features.size(0)))
        rep = logits.detach().to('cpu').numpy()

        init_centers, class_indices = calculate_class_centers_and_indices(rep,
                                                                          self.running_train_idx.detach().cpu().numpy(),
                                                                          self.labels[
                                                                              self.running_train_idx].detach().cpu().numpy())

        new_clusters_indices = list()
        xb = rep.astype('float32')
        for i in range(len(class_indices)):
            cluster_indices = class_indices[i]
            new_cluster_indices = list()
            for j in range(len(cluster_indices)):
                xq = np.expand_dims(rep[cluster_indices[j]], axis=0).astype('float32')
                index = faiss.IndexFlatL2(self.num_classes)
                index.add(xb)
                k = 4
                class_cluster_indices = np.squeeze(index.search(xq, k)[1], axis=0)[1: k]
                new_cluster_indices.append(class_cluster_indices)
            new_clusters_indices.append(new_cluster_indices)

        class_neighbor_indices_list = get_flatten(new_clusters_indices, self.num_classes)

        extended_clusters = merge_and_remove_duplicates(class_indices, class_neighbor_indices_list)
        extended_clusters = filter_nodes_by_confidence(rep, extended_clusters, self.p_max)

        self.labels, self.running_train_mask = update_labels_and_mask(self.running_train_mask, self.labels,
                                                                      extended_clusters, logits, self.num_classes)

    def train(self):
        for fold in range(self.args.folds):
            self.train_mask, self.val_mask, self.test_mask = masking(fold, self.data, self.args.label_rate)
            self._init_dataset()
            self._init_model()

            self.pretrain(fold, 0)
            self.adj_adjust(fold, 0)

            self._init_model()
            reset_parameters(self.model)

            for stage in range(self.args.stage):
                self.pretrain(fold, stage)
                self.self_train(fold, stage)
                self.adj_adjust(fold, stage)

            self._init_model()
            reset_parameters(self.model)
            for epoch in range(1, self.args.epochs + 1):
                if self.args.dataset != 'arxiv':
                    self.model.train()
                    self.optimizer.zero_grad()
                    logits, preds, logits_2, embeddings = self.model(self.features,
                                                                     edge_index2adj_t(self.running_edge_index,
                                                                                      self.features.size(0)),
                                                                     edge_index2adj_t(self.edge_index_2,
                                                                                      self.features.size(0)))
                    ce_loss_1 = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                    ce_loss_2 = F.cross_entropy(logits_2[self.running_train_mask], self.labels[self.running_train_mask])
                    x_dis_1 = get_feature_dis(logits)
                    x_dis_1_2 = get_distance_new(logits, logits_2)
                    nc_loss_1 = new_Ncontrast(x_dis_1, to_dense_adj(self.running_edge_index), x_dis_1_2)

                    x_dis_2 = get_feature_dis(logits_2)
                    x_dis_2_2 = get_distance_new(logits_2, logits)
                    nc_loss_2 = new_Ncontrast(x_dis_2, to_dense_adj(self.edge_index_2), x_dis_2_2)
                    total_loss = 0.5 * (ce_loss_1 + ce_loss_2) + 0.05 * (nc_loss_1 + nc_loss_2)

                    total_loss.backward()
                    self.optimizer.step()
                else:
                    self.model.train()
                    self.optimizer.zero_grad()
                    outs, preds, _, _ = self.model(self.features,
                                                   edge_index2adj_t(self.running_edge_index, self.features.size(0)),
                                                   edge_index2adj_t(self.edge_index_2, self.features.size(0)))
                    # print(out.shape)
                    loss = F.cross_entropy(outs[self.running_train_mask], self.labels[self.running_train_mask])
                    loss.backward()
                    self.optimizer.step()

                    for batch_size, n_id, adjs in self.running_train_loader_1:
                        self.optimizer.zero_grad()
                        out, preds, _, _ = self.model(self.features[n_id], adjs[0].to(self.device), None)
                        x_dis = get_feature_dis(out, self.device)
                        row, col = adjs[0]
                        adj = torch.zeros((n_id.size(0), n_id.size(0)), device=self.device)
                        adj[row, col] = 1
                        nc_loss = Ncontrast1(x_dis, adj)
                        nc_loss.backward()
                        self.optimizer.step()
                    for batch_size, n_id, adjs in self.running_train_loader_2:
                        self.optimizer.zero_grad()
                        _, preds, out, _ = self.model(self.features[n_id], None, adjs[0].to(self.device))
                        x_dis = get_feature_dis(out, self.device)
                        row, col = adjs[0]
                        adj = torch.zeros((n_id.size(0), n_id.size(0)), device=self.device)
                        adj[row, col] = 1
                        nc_loss_2 = Ncontrast1(x_dis, adj)
                        nc_loss_2.backward()
                        self.optimizer.step()
                if self.args.dataset != 'arxiv':
                    train_acc, val_acc, test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask,
                                                                    self.train_mask, self.val_mask,
                                                                    self.test_mask, self.device)
                else:
                    _, test_acc = ca(out, self.data.y, self.evaluator, self.running_val_idx, self.running_test_idx)

                st = '[Fold : {}][Epoch {}/{}] Test Accuracy: {:.4f}'.format(fold + 1, epoch + 1, 400, test_acc)

                self.evaluate(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)),
                              edge_index2adj_t(self.edge_index_2, self.features.size(0)), st)

            self.save_results(fold)
        self.summary()

    def update_edges_based_on_train_sets_mask(self, logits, class_indices, new_train_indices, stage, fold):
        # 计算扩充前训练集的模型预测结果
        model = DynamicEdgeIndexLearning(self.args, logits, self.data.x.size(0),
                                         new_train_indices, self.running_edge_index, self.device, self.num_classes)
        optimizer = Adam([model.edge_probs], lr=0.001)  # 只优化边概率

        # 使用log_softmax处理目标
        target = F.log_softmax(logits.detach(), dim=-1)
        kl = nn.KLDivLoss(reduction='batchmean')

        for e in range(400):
            optimizer.zero_grad()
            logit, new_edge_index = model.forward(self.features)
            loss = kl(logit, logits)
            loss.backward()

            optimizer.step()
            print('[Fold:{}/{}] [Stage:{}/{}] epoch {}/{} loss:{:.4f}'.format(fold + 1, self.args.folds, stage + 1,
                                                                              self.args.stage, e + 1, 400,
                                                                              loss.item()))
            if e % 3 == 0:
                print(f'Gradient of edge_probs at epoch {e}: {model.edge_probs.grad}')
        return new_edge_index.to(self.device)


class DynamicEdgeIndexLearning(nn.Module):
    def __init__(self, args, old_logits, num_nodes, l_list, old_edge_index, device, num_classes, min_prob=1e-6):
        super(DynamicEdgeIndexLearning, self).__init__()
        self.num_nodes = num_nodes
        self.logits = old_logits
        self.args = args
        self.device = device
        self.l_list = l_list
        self.old_edge_index = old_edge_index.to(device)
        self.tau = 1.0
        self.num_classes = num_classes

        # 初始化edge_probs为全连接边的概率参数
        self.edge_probs = nn.Parameter(
            torch.randn(sum(len(sublist) * (len(sublist) - 1) for sublist in self.l_list)),
            requires_grad=True
        )

    def _initial_model(self, x):
        # 确保GCN模型支持edge_weight参数
        self.model = GCN3(x.size(1), 256, self.num_classes).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        kl_2 = nn.KLDivLoss(reduction='batchmean')
        for epo in range(200):
            self.optimizer.zero_grad()
            logits, _, _ = self.model(x, self.old_edge_index)
            kl_loss = kl_2(logits, self.logits)
            kl_loss.backward()
            self.optimizer.step()
        torch.save(self.model.state_dict(), 'n2c_model_kl_{}.pth'.format(self.args.label_rate))
        self.model.load_state_dict(torch.load('n2c_model_kl_{}.pth'.format(self.args.label_rate), map_location='cpu'))

    def new_edges_generate(self):
        new_edges = []
        for cluster_indices in self.l_list:
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    new_edges.append([cluster_indices[i], cluster_indices[j]])
                    new_edges.append([cluster_indices[j], cluster_indices[i]])
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t().to(self.device)
        return new_edges_tensor

    def optimize_edge_index(self):
        # 使用sigmoid生成连续概率作为边的权重
        edge_weights = torch.sigmoid(self.edge_probs).to(self.device)
        new_edges = self.new_edges_generate()
        # 合并原始边和新边
        updated_edge_index = torch.cat((self.old_edge_index, new_edges), dim=1)
        # 原始边的权重为1，新边的权重为sigmoid后的概率
        old_weights = torch.ones(self.old_edge_index.size(1), device=self.device)
        updated_edge_weights = torch.cat((old_weights, edge_weights))
        return updated_edge_index, updated_edge_weights

    def forward(self, x):
        new_edge_index, new_edge_weights = self.optimize_edge_index()
        # 传递edge_weight到GCN模型
        self._initial_model(x)
        self.model.to(self.device)
        logits, _, _ = self.model(x, new_edge_index, edge_weight=new_edge_weights)
        return logits, new_edge_index

#