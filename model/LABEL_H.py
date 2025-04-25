from torch.optim import Adam
from copy import deepcopy
from embedders import embedder
from torch_geometric.utils import to_dense_adj
from data_process import *
import faiss
from src.utils import *
from layers.GNN import H2GCN
from embedders import eidx_to_sp
from scipy.sparse import coo_matrix

class CL_Trainer_H(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def _init_model(self):
        self.model = H2GCN(feat_dim=1433, hidden_dim=128, class_dim=self.num_classes).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        self.optimizer_n2c = Adam(self.model.parameters(), lr=self.args.lr * 2, weight_decay=self.args.decay)

    def _init_dataset(self):

        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)
        self.running_normalized_adj = deepcopy(self.normalized_adj)
        self.running_normalized_sys_adj = deepcopy(self.normalized_sys_adj)
        self.running_edge_index = deepcopy(self.edge_index)


        eta = self.data.num_nodes / (to_dense_adj(self.data.edge_index).sum() / self.data.num_nodes) ** len(
            self.hidden_layers)
        self.t = (self.labels[self.train_mask].unique(return_counts=True)[1] * 3 * eta / len(
            self.labels[self.train_mask])).type(torch.int64)
        self.t = self.t / self.args.stage

    def pretrain(self, mask, stage):
        reset_parameters(self.model)
        for epoch in range(600):
            self.model.train()
            self.optimizer.zero_grad()

            logits, logits_2, preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index), self.h2adj_2, self.features)
            ce_loss_1 = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
            ce_loss_2 = F.cross_entropy(logits_2[self.running_train_mask], self.labels[self.running_train_mask])

            x_dis_1 = get_feature_dis(logits)
            x_dis_1_2 = get_distance_new(logits, logits_2)
            nc_loss_1 = new_Ncontrast(x_dis_1, self.running_normalized_sys_adj, x_dis_1_2)

            x_dis_2 = get_feature_dis(logits_2)
            x_dis_2_2 = get_distance_new(logits_2, logits)
            nc_loss_2 = new_Ncontrast(x_dis_2, self.running_normalized_sys_adj, x_dis_2_2)

            total_loss = 0.5 * (ce_loss_1 + ce_loss_2) + 0.05 * (nc_loss_1 + nc_loss_2)

            total_loss.backward()
            self.optimizer.step()

            train_acc, val_acc, test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask,
                                                            self.train_mask, self.val_mask,
                                                            self.test_mask, self.device)

            st = '[Fold: {}/{}][Stage : {}/{}][Epoch {}/{}] Total Loss: {:.4f} Test Accuracy: {:.4f}'.format(mask+1, self.args.folds, stage + 1, self.args.stage, epoch + 1, 400, total_loss.item(), test_acc.item())

            print(st)

    def adj_adjust(self, mask, stage):
        self.model.eval()
        logits, logits_2, preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)
        detect_logits = logits.detach()
        embeddings = logits.detach().cpu().numpy()
        seed_indices = np.where(self.running_train_mask.detach().cpu().numpy())[0]
        seed_labels = self.labels[self.running_train_mask]
        init_centers, class_indices = get_centers(seed_indices, seed_labels, embeddings)

        logits_eval = logits.detach().cpu().numpy()
        new_clusters_indices = list()
        xb = logits_eval.astype('float32')
        for i in range(self.num_classes):
            cluster_indices = class_indices[i]
            new_cluster_indices = list()
            for j in range(len(cluster_indices)):
                xq = np.expand_dims(logits_eval[cluster_indices[j]], axis=0).astype('float32')
                index = faiss.IndexFlatL2(7)
                index.add(xb)
                k = self.args.edge_range
                class_cluster_indices = np.squeeze(index.search(xq, k)[1], axis=0)[1: k]
                new_cluster_indices.append(class_cluster_indices)
            new_clusters_indices.append(new_cluster_indices)

        class0_cluster_indices = np.array(new_clusters_indices[0]).ravel()
        class1_cluster_indices = np.array(new_clusters_indices[1]).ravel()
        class2_cluster_indices = np.array(new_clusters_indices[2]).ravel()
        class3_cluster_indices = np.array(new_clusters_indices[3]).ravel()
        class4_cluster_indices = np.array(new_clusters_indices[4]).ravel()
        class5_cluster_indices = np.array(new_clusters_indices[5]).ravel()
        class6_cluster_indices = np.array(new_clusters_indices[6]).ravel()

        # 检查是相互之间否有重复的节点序号，若有则删去，并与已知标签节点进行扩充
        extended_cluster0, extended_cluster1, extended_cluster2, extended_cluster3, extended_cluster4, extended_cluster5, \
            extended_cluster6 = get_extended_vectors_indices(class0_cluster_indices, class1_cluster_indices,
                                                             class2_cluster_indices, class3_cluster_indices,
                                                             class4_cluster_indices, class5_cluster_indices,
                                                             class6_cluster_indices, class_indices)

        # self-correction
        hc_extended_cluster0, hc_extended_cluster1, hc_extended_cluster2, hc_extended_cluster3, hc_extended_cluster4, \
            hc_extended_cluster5, hc_extended_cluster6 = self_correct(self.p_max, extended_cluster0, extended_cluster1,
                                                                      extended_cluster2, extended_cluster3,
                                                                      extended_cluster4, extended_cluster5,
                                                                      extended_cluster6, detect_logits)

        lists = list()
        lists.append(hc_extended_cluster0)
        lists.append(hc_extended_cluster1)
        lists.append(hc_extended_cluster2)
        lists.append(hc_extended_cluster3)
        lists.append(hc_extended_cluster4)
        lists.append(hc_extended_cluster5)
        lists.append(hc_extended_cluster6)
        # 自动修剪邻接矩阵
        self.running_normalized_sys_adj, self.running_edge_index = self.ada_edge(class_indices, hc_extended_cluster0, hc_extended_cluster1, hc_extended_cluster2, hc_extended_cluster3, hc_extended_cluster4, hc_extended_cluster5, hc_extended_cluster6, mask, stage)


    def self_train(self, mask, stage):
        # 寻找各自的最近邻（K=20）
        self.model.eval()
        logits, logits_2, preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)
        logits = logits.detach().cpu().numpy()

        seed_indices = np.where(self.running_train_mask.detach().cpu().numpy())[0]
        seed_labels = self.labels[self.running_train_mask]
        init_centers, class_indices = get_centers(seed_indices, seed_labels, logits)

        new_clusters_indices = list()

        xb = logits.astype('float32')
        for i in range(self.num_classes):
            cluster_indices = class_indices[i]
            new_cluster_indices = list()
            for j in range(len(cluster_indices)):
                xq = np.expand_dims(logits[cluster_indices[j]], axis=0).astype('float32')
                index = faiss.IndexFlatL2(7)
                index.add(xb)
                k = 6
                class_cluster_indices = np.squeeze(index.search(xq, k)[1], axis=0)[1: k]
                new_cluster_indices.append(class_cluster_indices)
            new_clusters_indices.append(new_cluster_indices)

        class0_neighbor_indices = np.array(new_clusters_indices[0]).ravel()
        class1_neighbor_indices = np.array(new_clusters_indices[1]).ravel()
        class2_neighbor_indices = np.array(new_clusters_indices[2]).ravel()
        class3_neighbor_indices = np.array(new_clusters_indices[3]).ravel()
        class4_neighbor_indices = np.array(new_clusters_indices[4]).ravel()
        class5_neighbor_indices = np.array(new_clusters_indices[5]).ravel()
        class6_neighbor_indices = np.array(new_clusters_indices[6]).ravel()
        class0_indices = class_indices[0]
        class1_indices = class_indices[1]
        class2_indices = class_indices[2]
        class3_indices = class_indices[3]
        class4_indices = class_indices[4]
        class5_indices = class_indices[5]
        class6_indices = class_indices[6]
        # print(len(class0_neighbor_indices) + len(class1_neighbor_indices) + len(class2_neighbor_indices) + len(class3_neighbor_indices) + len(class4_neighbor_indices) + len(class5_neighbor_indices) + len(class6_neighbor_indices))

        # self correct
        class0_neighbor_indices, class1_neighbor_indices, class2_neighbor_indices, class3_neighbor_indices, \
            class4_neighbor_indices, class5_neighbor_indices, class6_neighbor_indices = self_correct(self.p_max,
                                                                                                     class0_neighbor_indices,
                                                                                                     class1_neighbor_indices,
                                                                                                     class2_neighbor_indices,
                                                                                                     class3_neighbor_indices,
                                                                                                     class4_neighbor_indices,
                                                                                                     class5_neighbor_indices,
                                                                                                     class6_neighbor_indices,
                                                                                                     logits)

        print(len(class0_neighbor_indices) + len(class1_neighbor_indices) + len(class2_neighbor_indices) + len(
            class3_neighbor_indices) + len(class4_neighbor_indices) + len(class5_neighbor_indices) + len(
            class6_neighbor_indices))

        # 重置网络参数
        reset_parameters(self.model)

        for epoch in range(400):
            self.model.train()
            self.optimizer_n2c.zero_grad()
            logits, logits_2, preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)

            ce_loss = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])

            # N2N Loss 提前做好的采样
            contrastive_loss0 = N2N_MIM_Loss_2(logits, class0_indices, class0_neighbor_indices, class1_neighbor_indices,
                                               class2_neighbor_indices, class3_neighbor_indices,
                                               class4_neighbor_indices, class5_neighbor_indices,
                                               class6_neighbor_indices)
            contrastive_loss1 = N2N_MIM_Loss_2(logits, class1_indices, class1_neighbor_indices, class0_neighbor_indices,
                                               class2_neighbor_indices, class3_neighbor_indices,
                                               class4_neighbor_indices, class5_neighbor_indices,
                                               class6_neighbor_indices)
            contrastive_loss2 = N2N_MIM_Loss_2(logits, class2_indices, class2_neighbor_indices, class0_neighbor_indices,
                                               class1_neighbor_indices, class2_neighbor_indices,
                                               class4_neighbor_indices, class5_neighbor_indices,
                                               class6_neighbor_indices)
            contrastive_loss3 = N2N_MIM_Loss_2(logits, class3_indices, class3_neighbor_indices, class0_neighbor_indices,
                                               class1_neighbor_indices, class2_neighbor_indices,
                                               class4_neighbor_indices, class5_neighbor_indices,
                                               class6_neighbor_indices)
            contrastive_loss4 = N2N_MIM_Loss_2(logits, class4_indices, class4_neighbor_indices, class0_neighbor_indices,
                                               class1_neighbor_indices, class2_neighbor_indices,
                                               class3_neighbor_indices, class5_neighbor_indices,
                                               class6_neighbor_indices)
            contrastive_loss5 = N2N_MIM_Loss_2(logits, class5_indices, class5_neighbor_indices, class0_neighbor_indices,
                                               class1_neighbor_indices, class2_neighbor_indices,
                                               class3_neighbor_indices, class4_neighbor_indices,
                                               class6_neighbor_indices)
            contrastive_loss6 = N2N_MIM_Loss_2(logits, class6_indices, class6_neighbor_indices, class0_neighbor_indices,
                                               class1_neighbor_indices, class2_neighbor_indices,
                                               class3_neighbor_indices, class4_neighbor_indices,
                                               class5_neighbor_indices)
            contrastive_loss = (contrastive_loss0 + contrastive_loss1 + contrastive_loss2 + contrastive_loss3 +
                                contrastive_loss4 + contrastive_loss5 + contrastive_loss6) / 7

            total_loss = ce_loss + contrastive_loss

            total_loss.backward(retain_graph=True)
            self.optimizer_n2c.step()

            _, _, test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask, self.train_mask,
                                              self.val_mask, self.test_mask, self.device)

            st = '[Fold: {}/{}][Stage : {}/{}][Epoch {}/{}] CE Loss: {:.4f} N2C Loss: {:.4f} Loss: {:.4f} Test Accuracy: ' \
                 '{:.4f}'.format(mask+1, self.args.folds, stage + 1, self.args.stage, epoch + 1, 400, ce_loss.item(),
                                 contrastive_loss.item(), total_loss.item(), test_acc.item())

            print(st)

        # 进行训练集的扩充：邻居搜索 + 自我纠正
        self.model.eval()
        logits, logits_2, preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)
        rep = logits.detach().to('cpu').numpy()

        # Calculating initial centers
        seed_indices = np.where(self.running_train_mask.detach().cpu().numpy())[0]
        # print(seed_indices)
        seed_labels = self.labels[self.running_train_mask]
        init_centers, class_indices = get_centers(seed_indices, seed_labels, rep)

        new_clusters_indices = list()
        xb = rep.astype('float32')
        for i in range(len(class_indices)):
            cluster_indices = class_indices[i]
            new_cluster_indices = list()
            for j in range(len(cluster_indices)):
                xq = np.expand_dims(rep[cluster_indices[j]], axis=0).astype('float32')
                index = faiss.IndexFlatL2(7)
                index.add(xb)
                k = 4
                class_cluster_indices = np.squeeze(index.search(xq, k)[1], axis=0)[1: k]
                new_cluster_indices.append(class_cluster_indices)
            new_clusters_indices.append(new_cluster_indices)

        class0_cluster_indices = np.array(new_clusters_indices[0]).ravel()
        class1_cluster_indices = np.array(new_clusters_indices[1]).ravel()
        class2_cluster_indices = np.array(new_clusters_indices[2]).ravel()
        class3_cluster_indices = np.array(new_clusters_indices[3]).ravel()
        class4_cluster_indices = np.array(new_clusters_indices[4]).ravel()
        class5_cluster_indices = np.array(new_clusters_indices[5]).ravel()
        class6_cluster_indices = np.array(new_clusters_indices[6]).ravel()

        # 检查是相互之间否有重复的节点序号，若有则删去，并与已知标签节点进行扩充
        extended_cluster0, extended_cluster1, extended_cluster2, extended_cluster3, extended_cluster4, extended_cluster5, \
            extended_cluster6 = get_extended_vectors_indices(class0_cluster_indices, class1_cluster_indices,
                                                             class2_cluster_indices, class3_cluster_indices,
                                                             class4_cluster_indices, class5_cluster_indices,
                                                             class6_cluster_indices, class_indices)
        print(len(extended_cluster0) + len(extended_cluster1) + len(extended_cluster2) + len(extended_cluster3) + len(
            extended_cluster4) + len(extended_cluster5) + len(extended_cluster6))

        # self-correction
        logits = rep
        hc_extended_cluster0, hc_extended_cluster1, hc_extended_cluster2, hc_extended_cluster3, hc_extended_cluster4, \
            hc_extended_cluster5, hc_extended_cluster6 = self_correct(self.p_max, extended_cluster0, extended_cluster1,
                                                                      extended_cluster2, extended_cluster3,
                                                                      extended_cluster4, extended_cluster5,
                                                                      extended_cluster6, logits)
        print(len(hc_extended_cluster0) + len(hc_extended_cluster1) + len(hc_extended_cluster2) +
              len(hc_extended_cluster3) + len(hc_extended_cluster4) + len(hc_extended_cluster5) +
              len(hc_extended_cluster6))

        y_train, self.running_train_mask = self.self_training(hc_extended_cluster0, hc_extended_cluster1,
                                                              hc_extended_cluster2, hc_extended_cluster3,
                                                              hc_extended_cluster4, hc_extended_cluster5,
                                                              hc_extended_cluster6)

        # 使用生成的伪标签调整原labels
        self.labels[self.running_train_mask] = torch.argmax(y_train[self.running_train_mask], dim=1)
        purity, purity0, purity1, purity2, purity3, purity4, purity5, purity6 = calculate_purity(self.data.y,
                                                                                                 hc_extended_cluster0,
                                                                                                 hc_extended_cluster1,
                                                                                                 hc_extended_cluster2,
                                                                                                 hc_extended_cluster3,
                                                                                                 hc_extended_cluster4,
                                                                                                 hc_extended_cluster5,
                                                                                                 hc_extended_cluster6)
        print(purity, purity0, purity1, purity2, purity3, purity4, purity5, purity6)


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
                ada_best_acc = 0.
                self.model.train()
                self.optimizer.zero_grad()

                logits, logits_2, preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)
                ce_loss_1 = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                ce_loss_2 = F.cross_entropy(logits_2[self.running_train_mask], self.labels[self.running_train_mask])

                x_dis_1 = get_feature_dis(logits)
                x_dis_1_2 = get_distance_new(logits, logits_2)
                nc_loss_1 = new_Ncontrast(x_dis_1, self.running_normalized_sys_adj, x_dis_1_2)

                x_dis_2 = get_feature_dis(logits_2)
                x_dis_2_2 = get_distance_new(logits_2, logits)
                nc_loss_2 = new_Ncontrast(x_dis_2, self.running_normalized_sys_adj, x_dis_2_2)

                final_loss = 0.5 * (ce_loss_1 + ce_loss_2) + 0.05 * (nc_loss_1 + nc_loss_2)
                final_loss.backward()
                self.optimizer.step()

                train_acc, val_acc, ada_test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask,
                                                                    self.train_mask, self.val_mask,
                                                                    self.test_mask, self.device)

                st = '[Fold : {}][Epoch {}/{}] Loss: {:.4f} Test Accuracy: {:.4f}'.format(fold + 1, epoch + 1, 400,
                                                                                          final_loss.item(),
                                                                                          ada_test_acc.item())

                # evaluation
                self.evaluate(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features, st)

                if self.cnt == self.args.patience:
                    print("early stopping!")
                    break
            self.save_results(fold)
        self.summary()


    def self_training(self, cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6):
        y_train = F.one_hot(self.labels).float()
        y_train[~self.running_train_mask] = 0
        num_class = y_train.shape[1]
        assert len(self.t) >= num_class
        indicator = torch.zeros(self.train_mask.shape, dtype=torch.bool)
        indices_list = list()
        indices_list.append(cluster0)
        indices_list.append(cluster1)
        indices_list.append(cluster2)
        indices_list.append(cluster3)
        indices_list.append(cluster4)
        indices_list.append(cluster5)
        indices_list.append(cluster6)
        indices_list = [element for sublist in indices_list for element in sublist]
        #  为扩充后的训练集生成mask
        indicator[indices_list] = True
        # torch.logical_and表示只有都是True才为True，其余均为False
        # 此时indicator中为True只有是与原标签不同的扩充节点
        indicator = torch.logical_and(torch.logical_not(self.running_train_mask), indicator.to(self.device))
        prediction = torch.zeros(self.data.x.shape[0], self.num_classes).to(self.device)

        # 为扩增后的节点生成one-hot编码，其余的都为0
        new_gcn_index = torch.zeros(self.data.num_nodes, dtype=torch.int64)
        for index in cluster0:
            new_gcn_index[index] = torch.tensor(0, dtype=torch.int64)
        for index in cluster1:
            new_gcn_index[index] = torch.tensor(1, dtype=torch.int64)
        for index in cluster2:
            new_gcn_index[index] = torch.tensor(2, dtype=torch.int64)
        for index in cluster3:
            new_gcn_index[index] = torch.tensor(3, dtype=torch.int64)
        for index in cluster4:
            new_gcn_index[index] = torch.tensor(4, dtype=torch.int64)
        for index in cluster5:
            new_gcn_index[index] = torch.tensor(5, dtype=torch.int64)
        for index in cluster6:
            new_gcn_index[index] = torch.tensor(6, dtype=torch.int64)
        prediction[torch.arange(self.data.num_nodes), new_gcn_index] = 1.0

        y_train = deepcopy(y_train)
        train_mask = deepcopy(self.running_train_mask)
        # train mask现在是为扩增后的训练集生成的mask
        train_mask[indicator] = 1
        # y_train现在为新增节点生成了伪标签
        y_train[indicator] = prediction[indicator]
        return y_train, train_mask

    def ada_edge(self, class_indices, cluster0_indices, cluster1_indices, cluster2_indices,
                 cluster3_indices, cluster4_indices, cluster5_indices, cluster6_indices, fold, stage):
        class_indices_list = [element for sublist in class_indices for element in sublist]
        # 未调整前的模型预测结果
        self.model.eval()
        logits, logits_2, preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()),
                                             self.h2adj_2, self.features)

        old_clusters_scores = list()
        for num in range(len(class_indices)):
            cluster = class_indices[num]
            for index in cluster:
                old_clusters_scores.append(torch.max(logits.detach().cpu()[index]))
        old_clusters_scores = np.array(old_clusters_scores)

        clusters_indices = list()
        clusters_indices.append(cluster0_indices)
        clusters_indices.append(cluster1_indices)
        clusters_indices.append(cluster2_indices)
        clusters_indices.append(cluster3_indices)
        clusters_indices.append(cluster4_indices)
        clusters_indices.append(cluster5_indices)
        clusters_indices.append(cluster6_indices)
        self.model.eval()
        for num in range(len(clusters_indices)):
            cluster_indices = clusters_indices[num]
            for i in range(len(cluster_indices)):
                for j in range(i, len(cluster_indices)):
                    if i != j:
                        # add edges
                        index_z = torch.all(
                            self.running_edge_index.t() == torch.tensor([cluster_indices[i], cluster_indices[j]]).to(
                                self.device), dim=1).nonzero(as_tuple=False)
                        index_f = torch.all(
                            self.running_edge_index.t() == torch.tensor([cluster_indices[j], cluster_indices[i]]).to(
                                self.device), dim=1).nonzero(as_tuple=False)
                        if index_z.nelement() > 0:
                            self.running_edge_index = torch.cat(
                                (self.running_edge_index.t()[:index_z], self.running_edge_index.t()[index_z + 1:])).t()
                            self.running_coo_matrix = coo_matrix((torch.ones(self.running_edge_index.shape[1]), (
                                self.running_edge_index.detach().cpu()[0], self.running_edge_index.detach().cpu()[1])),
                                                                 shape=(self.data.num_nodes, self.data.num_nodes))
                            self.running_normalized_adj = normalize_adj_rw(self.running_coo_matrix + coo_matrix((
                                torch.ones(
                                    self.data.num_nodes),
                                (
                                    [p for p
                                     in
                                     range(
                                         self.data.num_nodes)],
                                    [p for p
                                     in
                                     range(
                                         self.data.num_nodes)])),
                                shape=(
                                    self.data.num_nodes,
                                    self.data.num_nodes))).to(
                                self.device)

                            new_logits, _, new_preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)
                            new_clusters_scores = list()
                            for num_n in range(len(class_indices)):
                                cluster = class_indices[num_n]
                                for index in cluster:
                                    new_clusters_scores.append(torch.max(new_logits.detach().cpu()[index]))
                            new_clusters_scores = np.array(new_clusters_scores)
                            if len([s for s in (new_clusters_scores - old_clusters_scores) if s < 0]) > 0 and sum(
                                    new_preds[self.running_train_mask] - preds[self.running_train_mask]) != 0:
                                self.running_edge_index = torch.cat((self.running_edge_index.t(), torch.tensor(
                                    [cluster_indices[i], cluster_indices[j]]).unsqueeze(0).to(self.device))).t()
                                self.running_coo_matrix = coo_matrix((torch.ones(self.running_edge_index.shape[1]), (
                                    self.running_edge_index.detach().cpu()[0],
                                    self.running_edge_index.detach().cpu()[1])),
                                                                     shape=(self.data.num_nodes, self.data.num_nodes))

                        elif index_f.nelement() > 0:
                            self.running_edge_index = torch.cat(
                                (self.running_edge_index.t()[:index_f], self.running_edge_index.t()[index_f + 1:])).t()
                            self.running_coo_matrix = coo_matrix((torch.ones(self.running_edge_index.shape[1]), (
                                self.running_edge_index.detach().cpu()[0], self.running_edge_index.detach().cpu()[1])),
                                                                 shape=(self.data.num_nodes, self.data.num_nodes))
                            self.running_normalized_adj = normalize_adj_rw(self.running_coo_matrix + coo_matrix((
                                torch.ones(
                                    self.data.num_nodes),
                                (
                                    [p for p
                                     in
                                     range(
                                         self.data.num_nodes)],
                                    [p for p
                                     in
                                     range(
                                         self.data.num_nodes)])),
                                shape=(
                                    self.data.num_nodes,
                                    self.data.num_nodes))).to(
                                self.device)

                            new_logits, _, new_preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)
                            new_clusters_scores = list()
                            for num_n in range(len(class_indices)):
                                cluster = class_indices[num_n]
                                for index in cluster:
                                    new_clusters_scores.append(torch.max(new_logits.detach().cpu()[index]))
                            new_clusters_scores = np.array(new_clusters_scores)
                            if len([s for s in (new_clusters_scores - old_clusters_scores) if s < 0]) > 0 and sum(
                                    new_preds[self.running_train_mask] - preds[self.running_train_mask]) != 0:
                                self.running_edge_index = torch.cat((self.running_edge_index.t(), torch.tensor(
                                    [cluster_indices[i], cluster_indices[j]]).unsqueeze(0).to(self.device))).t()
                                self.running_coo_matrix = coo_matrix((torch.ones(self.running_edge_index.shape[1]), (
                                    self.running_edge_index.detach().cpu()[0],
                                    self.running_edge_index.detach().cpu()[1])),
                                                                     shape=(self.data.num_nodes, self.data.num_nodes))

                        else:
                            self.running_edge_index = torch.cat((self.running_edge_index.t(), torch.tensor(
                                [cluster_indices[i], cluster_indices[j]]).unsqueeze(0).to(self.device))).t()
                            self.running_coo_matrix = coo_matrix((torch.ones(self.running_edge_index.shape[1]), (
                                self.running_edge_index.detach().cpu()[0], self.running_edge_index.detach().cpu()[1])),
                                                                 shape=(self.data.num_nodes, self.data.num_nodes))
                            self.running_normalized_adj = normalize_adj_rw(self.running_coo_matrix + coo_matrix((
                                torch.ones(
                                    self.data.num_nodes),
                                (
                                    [p for p
                                     in
                                     range(
                                         self.data.num_nodes)],
                                    [p for p
                                     in
                                     range(
                                         self.data.num_nodes)])),
                                shape=(
                                    self.data.num_nodes,
                                    self.data.num_nodes))).to(
                                self.device)
                            new_logits, _, new_preds = self.model(eidx_to_sp(len(self.features), self.running_edge_index.detach().cpu()), self.h2adj_2, self.features)
                            new_clusters_scores = list()
                            for num_n in range(len(class_indices)):
                                cluster = class_indices[num_n]
                                for index in cluster:
                                    new_clusters_scores.append(torch.max(new_logits.detach().cpu()[index]))
                            new_clusters_scores = np.array(new_clusters_scores)
                            if len([s for s in (new_clusters_scores - old_clusters_scores) if s < 0]) > 0 and sum(
                                    new_preds[self.running_train_mask] - preds[self.running_train_mask]) != 0:
                                self.running_edge_index = self.running_edge_index.t()[:-1].t()
                                self.running_coo_matrix = coo_matrix((torch.ones(self.running_edge_index.shape[1]), (
                                    self.running_edge_index.detach().cpu()[0],
                                    self.running_edge_index.detach().cpu()[1])),
                                                                     shape=(self.data.num_nodes, self.data.num_nodes))
                    else:
                        continue

        normalized_adj = deepcopy(normalize_adj_rw(self.running_coo_matrix + coo_matrix((
                                                                                        torch.ones(self.data.num_nodes),
                                                                                        ([p for p in
                                                                                          range(self.data.num_nodes)],
                                                                                         [p for p in
                                                                                          range(self.data.num_nodes)])),
                                                                                        shape=(self.data.num_nodes,
                                                                                               self.data.num_nodes))).to(
            self.device))
        edge_index = self.running_coo_matrix + coo_matrix((torch.ones(self.data.num_nodes), (
        [p for p in range(self.data.num_nodes)], [p for p in range(self.data.num_nodes)])),
                                                          shape=(self.data.num_nodes, self.data.num_nodes))
        edge_index = torch.tensor(np.vstack((edge_index.nonzero())), dtype=torch.long)
        return normalized_adj.to(self.device), edge_index.to(self.device)


