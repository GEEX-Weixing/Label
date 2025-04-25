from torch.optim import Adam
from copy import deepcopy
from embedder import embedder
from torch_geometric.utils import to_dense_adj
from data_process import *
import faiss
from src.utils import *
from layers.GNN import GCN
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import to_scipy_sparse_matrix

class CL_Trainer_C(embedder):
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
        self.running_train_loader_1 = NeighborSampler(self.running_edge_index, node_idx=self.running_train_idx, sizes=[-1], batch_size=1024, shuffle=True, num_workers=12)
        self.running_train_loader_2 = NeighborSampler(self.edge_index_2, node_idx=self.running_train_idx,  sizes=[-1], batch_size=1024, shuffle=True, num_workers=12)
        eta = self.data.num_nodes / (to_dense_adj(self.data.edge_index).sum() / self.data.num_nodes) ** len(self.hidden_layers)
        self.t = (self.labels[self.train_mask].unique(return_counts=True)[1] * 3 * eta / len(self.labels[self.train_mask])).type(torch.int64)
        self.t = self.t / self.args.stage

    def pretrain(self, mask, stage):
        reset_parameters(self.model)
        for epoch in range(400):
            if self.args.dataset != 'arxiv':
                self.model.train()
                self.optimizer.zero_grad()
                logits, preds, logits_2, embeddings = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
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
                outs, preds, _, _ = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
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

            st = '[Fold: {}/{}][Stage : {}/{}][Epoch {}/{}] Test Accuracy: {:.4f}'.format(mask+1, self.args.folds, stage + 1, self.args.stage, epoch + 1, 400, test_acc.item())

            print(st)
    def adj_adjust(self, mask, stage):
        self.model.eval()
        logits, preds, _, embeddings = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
        detect_logits = logits.detach().cpu().numpy()
        init_centers, class_indices = calculate_class_centers_and_indices(detect_logits, self.running_train_idx.detach().cpu().numpy(), self.labels[self.running_train_idx].detach().cpu().numpy())

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
        self.running_edge_index = self.update_edges_based_on_train_sets(class_indices, extended_clusters)


    def self_train(self, mask, stage):
            # 寻找各自的最近邻（K=20）
            self.model.eval()
            logits, preds, _, embeddings = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
            logits = logits.detach().cpu().numpy()

            init_centers, class_indices = calculate_class_centers_and_indices(logits, self.running_train_idx.detach().cpu().numpy(), self.labels[self.running_train_idx].detach().cpu().numpy())

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
            for epoch in range(400):
                self.model.train()
                self.optimizer_n2c.zero_grad()
                logits, preds, logits_2, x = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))

                ce_loss = F.cross_entropy(logits[self.running_train_idx], self.labels[self.running_train_idx])
                contrastive_loss = calculate_loss(class_indices, class_neighbor_indices_list, logits, self.num_classes)
                # N2N Loss 提前做好的采样

                total_loss = ce_loss + contrastive_loss
                # total_loss.requires_grad_(True)

                total_loss.backward(retain_graph=True)
                self.optimizer_n2c.step()
                if self.args.dataset !='arxiv':
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
            logits, preds, _, _ = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
            rep = logits.detach().to('cpu').numpy()

            init_centers, class_indices = calculate_class_centers_and_indices(rep, self.running_train_idx.detach().cpu().numpy(), self.labels[self.running_train_idx].detach().cpu().numpy())

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

            self.labels, self.running_train_mask = update_labels_and_mask(self.running_train_mask, self.labels, extended_clusters, logits, self.num_classes)

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
            for epoch in range(1, self.args.epochs+1):
                if self.args.dataset != 'arxiv':
                    self.model.train()
                    self.optimizer.zero_grad()
                    logits, preds, logits_2, embeddings = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
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
                    outs, preds, _, _ = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
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

                self.evaluate(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)), st)

            self.save_results(fold)
        self.summary()

    def update_edges_based_on_train_sets(self, class_indices, new_train_indices):
        # 计算扩充前训练集的模型预测结果
        self.model.eval()
        logits, preds, _, embeddings = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))

        # 保存扩充前训练集的scores
        old_clusters_scores = []
        for class_idx in class_indices:
            for index in class_idx:
                old_clusters_scores.append(torch.max(logits.detach().cpu()[index]))
        old_clusters_scores = np.array(old_clusters_scores)

        # 合并扩充前和扩充后的训练集索引
        all_train_indices = class_indices + new_train_indices

        # 遍历扩充后的训练集，检查每对节点
        for num in range(len(new_train_indices)):
            cluster_indices = new_train_indices[num]
            for i in range(len(cluster_indices)):
                for j in range(i, len(cluster_indices)):
                    if i != j:
                        # 处理每一对扩充后的节点
                        index_z = torch.all(self.running_edge_index.t() == torch.tensor([cluster_indices[i], cluster_indices[j]]).to(self.device), dim=1).nonzero(as_tuple=False)
                        index_f = torch.all(self.running_edge_index.t() == torch.tensor([cluster_indices[j], cluster_indices[i]]).to(self.device), dim=1).nonzero(as_tuple=False)

                        if index_z.nelement() > 0:  # 如果已经存在边
                            # 删除已有边
                            self.running_edge_index = torch.cat((self.running_edge_index.t()[:index_z], self.running_edge_index.t()[index_z + 1:])).t()

                            # 重新计算新的模型预测结果
                            new_logits, new_preds, _, new_embeddings = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
                            new_clusters_scores = []
                            for class_idx in class_indices:
                                for index in class_idx:
                                    new_clusters_scores.append(torch.max(new_logits.detach().cpu()[index]))
                            new_clusters_scores = np.array(new_clusters_scores)

                            # 比较模型输出的变化
                            if len([s for s in (new_clusters_scores - old_clusters_scores) if s < 0]) > 0 and sum(new_preds[self.running_train_mask] - preds[self.running_train_mask]) != 0:
                                # 更新边
                                self.running_edge_index = torch.cat((self.running_edge_index.t(), torch.tensor([cluster_indices[i], cluster_indices[j]]).unsqueeze(0).to(self.device))).t()

                        elif index_f.nelement() > 0:  # 如果该对节点是反向边
                            # 删除反向边
                            self.running_edge_index = torch.cat((self.running_edge_index.t()[:index_f], self.running_edge_index.t()[index_f + 1:])).t()

                            # 更新后的计算
                            new_logits, new_preds, _, new_embeddings = self.model(self.features, edge_index2adj_t(self.running_edge_index, self.features.size(0)), edge_index2adj_t(self.edge_index_2, self.features.size(0)))
                            new_clusters_scores = []
                            for class_idx in class_indices:
                                for index in class_idx:
                                    new_clusters_scores.append(torch.max(new_logits.detach().cpu()[index]))
                            new_clusters_scores = np.array(new_clusters_scores)
                            if len([s for s in (new_clusters_scores - old_clusters_scores) if s < 0]) > 0 and sum(new_preds[self.running_train_mask] - preds[self.running_train_mask]) != 0:
                                self.running_edge_index = torch.cat((self.running_edge_index.t(), torch.tensor([cluster_indices[i], cluster_indices[j]]).unsqueeze(0).to(self.device))).t()

                        else:  # 处理未存在的边
                            self.running_edge_index = torch.cat((self.running_edge_index.t(), torch.tensor([cluster_indices[i], cluster_indices[j]]).unsqueeze(0).to(self.device))).t()

        return self.running_edge_index

