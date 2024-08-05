from torch.optim import Adam
from copy import deepcopy
from embedder import embedder
from torch_geometric.utils import to_dense_adj
from data_process import *
import faiss
from src.utils import *
from layers.GNN import GCN


class CL_Trainer_M(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def _init_model(self):

        self.model = GCN(self.normalized_sys_adj, self.adj_2, 1433, 256, self.num_classes).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        self.optimizer_n2c = Adam(self.model.parameters(), lr=self.args.lr * 2, weight_decay=self.args.decay)

    def _init_dataset(self):

        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)
        self.running_normalized_adj = deepcopy(self.normalized_adj)
        self.running_normalized_sys_adj = deepcopy(self.normalized_sys_adj)
        self.running_dense_adj = deepcopy(self.dense_adj)

        eta = self.data.num_nodes / (to_dense_adj(self.data.edge_index).sum() / self.data.num_nodes) ** len(
            self.hidden_layers)
        self.t = (self.labels[self.train_mask].unique(return_counts=True)[1] * 3 * eta / len(
            self.labels[self.train_mask])).type(torch.int64)
        self.t = self.t / self.args.stage

    def pretrain(self, mask, stage):
        reset_parameters(self.model)
        best_val_acc = 0.0
        for epoch in range(600):
            self.model.train()
            self.optimizer.zero_grad()

            logits, preds, logits_2, embeddings = self.model(self.features)
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

            st = '[Fold: {}/{}][Stage : {}/{}][Epoch {}/{}] Total Loss: {:.4f} Test Accuracy: {:.4f}'.format(mask + 1,
                                                                                                             self.args.folds,
                                                                                                             stage + 1,
                                                                                                             self.args.stage,
                                                                                                             epoch + 1,
                                                                                                             600,
                                                                                                             total_loss.item(),
                                                                                                             test_acc.item())


        # np.save('M_ADE_NC_logits_{}_{}.npy'.format(mask+1, stage+1), logits.detach().cpu().numpy())
        # np.save('M_ADE_NC_embeddings_{}_{}.npy'.format(mask + 1, stage + 1), embeddings.detach().cpu().numpy())

        # seed_indices = np.where(self.running_train_mask.detach().cpu().numpy())[0]
        # seed_labels = self.labels[self.running_train_mask]
        # init_centers, class_indices = get_centers(seed_indices, seed_labels, embeddings.detach().cpu().numpy())
        # np.save('running_train_mask_class_indices/M_ADE/NC_labeled_nodes_class_indices_{}_{}_class0.npy'.format(mask+1, stage+1),np.array(class_indices[0]))
        # np.save('running_train_mask_class_indices/M_ADE/NC_labeled_nodes_class_indices_{}_{}_class1.npy'.format(mask+1, stage+1),np.array(class_indices[1]))
        # np.save('running_train_mask_class_indices/M_ADE/NC_labeled_nodes_class_indices_{}_{}_class2.npy'.format(mask+1, stage+1),np.array(class_indices[2]))
        # np.save('running_train_mask_class_indices/M_ADE/NC_labeled_nodes_class_indices_{}_{}_class3.npy'.format(mask+1, stage+1),np.array(class_indices[3]))
        # np.save('running_train_mask_class_indices/M_ADE/NC_labeled_nodes_class_indices_{}_{}_class4.npy'.format(mask+1, stage+1),np.array(class_indices[4]))
        # np.save('running_train_mask_class_indices/M_ADE/NC_labeled_nodes_class_indices_{}_{}_class5.npy'.format(mask+1, stage+1),np.array(class_indices[5]))
        # np.save('running_train_mask_class_indices/M_ADE/NC_labeled_nodes_class_indices_{}_{}_class6.npy'.format(mask+1, stage+1),np.array(class_indices[6]))

    def adj_adjust(self, mask, stage):
        self.model.eval()
        logits, preds, _, embeddings = self.model(self.features)
        detect_logits = logits.detach()
        embeddings = embeddings.detach().cpu().numpy()
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

        # np.save('M_ADE_edges_range_nodes_class_indices/M_ADE_nodes_class_indices_{}_{}_class0.npy'.format(mask + 1, stage + 1), np.array(hc_extended_cluster0))
        # np.save('M_ADE_edges_range_nodes_class_indices/M_ADE_nodes_class_indices_{}_{}_class1.npy'.format(mask + 1, stage + 1), np.array(hc_extended_cluster1))
        # np.save('M_ADE_edges_range_nodes_class_indices/M_ADE_nodes_class_indices_{}_{}_class2.npy'.format(mask + 1, stage + 1), np.array(hc_extended_cluster2))
        # np.save('M_ADE_edges_range_nodes_class_indices/M_ADE_nodes_class_indices_{}_{}_class3.npy'.format(mask + 1, stage + 1), np.array(hc_extended_cluster3))
        # np.save('M_ADE_edges_range_nodes_class_indices/M_ADE_nodes_class_indices_{}_{}_class4.npy'.format(mask + 1, stage + 1), np.array(hc_extended_cluster4))
        # np.save('M_ADE_edges_range_nodes_class_indices/M_ADE_nodes_class_indices_{}_{}_class5.npy'.format(mask + 1, stage + 1), np.array(hc_extended_cluster5))
        # np.save('M_ADE_edges_range_nodes_class_indices/M_ADE_nodes_class_indices_{}_{}_class6.npy'.format(mask + 1, stage + 1), np.array(hc_extended_cluster6))

        # 自动修剪邻接矩阵
        # self.running_normalized_adj, self.running_dense_adj, self.running_normalized_sys_adj = self.ada_edge(class_indices, hc_extended_cluster0, hc_extended_cluster1, hc_extended_cluster2, hc_extended_cluster3, hc_extended_cluster4, hc_extended_cluster5, hc_extended_cluster6, mask, stage)
        self.running_normalized_adj, self.running_dense_adj, self.running_normalized_sys_adj = self.ada_edge_mask(stage,
                                                                                                                  logits,
                                                                                                                  hc_extended_cluster0,
                                                                                                                  hc_extended_cluster1,
                                                                                                                  hc_extended_cluster2,
                                                                                                                  hc_extended_cluster3,
                                                                                                                  hc_extended_cluster4,
                                                                                                                  hc_extended_cluster5,
                                                                                                                  hc_extended_cluster6,
                                                                                                                  mask)

    def self_train(self, mask, stage):
        # 寻找各自的最近邻（K=20）
        self.model.eval()
        logits, _, _, _ = self.model(self.features)
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
                k = 4
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

        # np.save('NC-N2C/M_ADE/NC-N2C_neighbors_{}_{}_class0.npy'.format(mask+1, stage+1), np.array(class0_neighbor_indices))
        # np.save('NC-N2C/M_ADE/NC-N2C_neighbors_{}_{}_class1.npy'.format(mask+1, stage+1), np.array(class1_neighbor_indices))
        # np.save('NC-N2C/M_ADE/NC-N2C_neighbors_{}_{}_class2.npy'.format(mask+1, stage+1), np.array(class2_neighbor_indices))
        # np.save('NC-N2C/M_ADE/NC-N2C_neighbors_{}_{}_class3.npy'.format(mask+1, stage+1), np.array(class3_neighbor_indices))
        # np.save('NC-N2C/M_ADE/NC-N2C_neighbors_{}_{}_class4.npy'.format(mask+1, stage+1), np.array(class4_neighbor_indices))
        # np.save('NC-N2C/M_ADE/NC-N2C_neighbors_{}_{}_class5.npy'.format(mask+1, stage+1), np.array(class5_neighbor_indices))
        # np.save('NC-N2C/M_ADE/NC-N2C_neighbors_{}_{}_class6.npy'.format(mask+1, stage+1), np.array(class6_neighbor_indices))

        # 重置网络参数
        reset_parameters(self.model)
        best_val_acc = 0.

        for epoch in range(400):
            self.model.train()
            self.optimizer_n2c.zero_grad()
            logits, preds, _, embeddings = self.model(self.features)

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

            _, val_acc, test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask, self.train_mask,
                                                    self.val_mask, self.test_mask, self.device)

            st = '[Fold: {}/{}][Stage : {}/{}][Epoch {}/{}] CE Loss: {:.4f} N2C Loss: {:.4f} Loss: {:.4f} Test Accuracy: ' \
                 '{:.4f}'.format(mask + 1, self.args.folds, stage + 1, self.args.stage, epoch + 1, 400, ce_loss.item(),
                                 contrastive_loss.item(), total_loss.item(), test_acc.item())

            print(st)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'n2c_model_kl_{}.pth'.format(self.args.label_rate))

        # np.save('M_ADE_N2C_logits_{}_{}.npy'.format(mask + 1, stage + 1), logits.detach().cpu().numpy())
        # np.save('M_ADE_N2C_embeddings_{}_{}.npy'.format(mask + 1, stage + 1), embeddings.detach().cpu().numpy())

        # 进行训练集的扩充：邻居搜索 + 自我纠正
        self.model.eval()
        logits, _, _, _ = self.model(self.features)
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
        # print(sum(self.running_train_mask))
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

            for stage in range(self.args.stage):
                self.pretrain(fold, stage)

                self.self_train(fold, stage)
                self.adj_adjust(fold, stage)

            self._init_model()
            reset_parameters(self.model)
            for epoch in range(1, self.args.epochs + 1):
                ada_best_acc = 0.
                self.model.train()
                self.optimizer.zero_grad()

                logits, preds, logits_2, embeddings = self.model(self.features)
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

                st = '[Fold : {}][Epoch {}/{}] Loss: {:.4f} Test Accuracy: {:.4f}'.format(fold + 1, epoch + 1, 800,
                                                                                          final_loss.item(),
                                                                                          ada_test_acc.item())

                # evaluation
                self.evaluate(self.features, st)

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

    def ada_edge_mask(self, stage, logits_old, cluster0_indices, cluster1_indices, cluster2_indices, cluster3_indices,
                      cluster4_indices, cluster5_indices, cluster6_indices, fold):
        clusters_indices = list()
        clusters_indices.append(cluster0_indices)
        clusters_indices.append(cluster1_indices)
        clusters_indices.append(cluster2_indices)
        clusters_indices.append(cluster3_indices)
        clusters_indices.append(cluster4_indices)
        clusters_indices.append(cluster5_indices)
        clusters_indices.append(cluster6_indices)

        model = Mask(self.args, self.running_dense_adj, clusters_indices, self.device)
        kl = nn.KLDivLoss(reduction='batchmean')
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        for epoch in range(400):
            optimizer.zero_grad()
            logits, new_adj = model.forward(self.features)
            loss = kl(logits, logits_old)
            loss.backward(retain_graph=True)

            optimizer.step()
            print('[Fold:{}/{}] [Stage:{}/{}] epoch {}/{}loss:{:.4f}'.format(fold + 1, self.args.folds, stage + 1,
                                                                             self.args.stage, epoch + 1, 400,
                                                                             loss.item()))
        normalized_sys_adj = deepcopy(normalize_adj(new_adj.detach().cpu()))
        normalized_adj = deepcopy(normalize_adj_rw(new_adj.detach().cpu()))
        dense_adj = deepcopy(new_adj.detach().cpu())
        # np.save('M_ADE_dense_adj_{}_{}_Label_rate-{}.npy'.format(fold + 1, stage + 1, self.args.label_rate), dense_adj)
        return normalized_adj.to(self.device), dense_adj.to(self.device), normalized_sys_adj.to(self.device)


class Mask(nn.Module):
    def __init__(self, args, run_dense_adj, clusters_indices, device):
        super(Mask, self).__init__()
        self.args = args
        self.device = device
        self.mask = nn.Parameter(torch.rand(2708, 2708, dtype=torch.float64).cuda()).to(device)
        self.register_parameter("masks", self.mask)
        self.run_dense_adj = run_dense_adj
        self.clusters_indices = clusters_indices

    def _initial_model(self):
        self.model = GCN(self.running_n_adj, self.running_n_adj, 1433, 256, 7)
        self.model.load_state_dict(torch.load('n2c_model_kl_{}.pth'.format(self.args.label_rate), map_location='cpu'))

    def ada_mask(self):
        x = torch.zeros(2708, 2708, dtype=torch.float32).to(self.device)
        for num in range(len(self.clusters_indices)):
            cluster_indices = self.clusters_indices[num]
            for i in range(len(cluster_indices)):
                for j in range(i, len(cluster_indices)):
                    x[cluster_indices[i]][cluster_indices[j]] = 1
                    x[cluster_indices[j]][cluster_indices[i]] = 1
        x = x.double().to(self.device)
        y = torch.triu(torch.ones(2708, 2708, dtype=torch.float32).cuda())
        a = self.mask
        a_softmax = F.sigmoid(a)
        with torch.no_grad():
            ind_sample = torch.bernoulli(a_softmax).bool()
            epsilon = torch.where(ind_sample, 1 - a_softmax, - a_softmax)
        b = a_softmax + epsilon
        c = b * (1 - y)
        d = c + c.T
        e = d * x
        new_adj = self.run_dense_adj * (1 - x) + e
        self.running_n_adj = deepcopy(normalize_adj_rw(new_adj.detach().cpu().numpy()))
        self.running_n_adj = self.running_n_adj.cuda()
        return new_adj

    def forward(self, x):
        # self.model.eval()

        new_adj = self.ada_mask()
        self._initial_model()
        self.model.to(self.device)
        logits, _, _, _ = self.model(x)
        return logits, new_adj
