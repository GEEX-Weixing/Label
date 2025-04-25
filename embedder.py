from src.get_data import get_datas
from src.utils import config2string
from src.utils import compute_accuracy
from data_process import *
from torch_geometric.utils import dropout_adj
from ogb.nodeproppred import Evaluator
from torch_geometric.utils import to_scipy_sparse_matrix
class embedder:
    def __init__(self, args):
        self.args = args

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        # self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        # torch.cuda.set_device(self.device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))

        # dataset
        self.data = get_datas(args.dataset)
        self.edge_index = self.data.edge_index
        self.edge_index_2 = dropout_adj(self.edge_index, p=0.5)[0]
        # self.adj_t = self.data.adj_t
        self.adj_t_2 = to_scipy_sparse_matrix(self.edge_index_2)
        self.features = preprocess_features(self.data.x.cpu()).to(self.device)
        self.evaluator = Evaluator(name='ogbn-arxiv')

        # Encoder
        self.hidden_layers = eval(args.layers)
        input_dim = self.data.x.size(1)
        rep_size = self.hidden_layers[-1]

        self.unique_labels = self.data.y.unique()
        self.num_classes = len(self.unique_labels)
        self.p_max = args.pmax



        # For evaluation
        self.best_val = 0
        self.epoch_list = []

        self.train_accs = []
        self.valid_accs = []
        self.test_accs = []
        self.running_train_accs = []
        self.running_valid_accs = []
        self.running_test_accs = []

    def evaluate(self, batch_data, adj_t, adj_t_2, st):

        # Classifier Accuracy
        self.model.eval()
        logits, preds, _, embeddings = self.model(batch_data, adj_t, adj_t_2)

        train_acc, val_acc, test_acc = compute_accuracy(preds, self.data.y, self.running_train_mask, self.train_mask, self.val_mask,
                                                        self.test_mask, self.device)
        self.running_train_accs.append(train_acc)
        self.running_valid_accs.append(val_acc)
        self.running_test_accs.append(test_acc)

        if val_acc > self.best_val:
            self.best_val = val_acc
            self.cnt = 0
        else:
            self.cnt += 1

        st += '| train_acc: {:.2f} | valid_acc : {:.2f} | test_acc : {:.2f} '.format(train_acc, val_acc, test_acc)
        print(st)

    def save_results(self, fold):

        train_acc, val_acc, test_acc = torch.tensor(self.running_train_accs), torch.tensor(
            self.running_valid_accs), torch.tensor(self.running_test_accs)
        selected_epoch = val_acc.argmax()

        best_train_acc = train_acc[selected_epoch]
        best_val_acc = val_acc[selected_epoch]
        best_test_acc = test_acc[selected_epoch]

        self.epoch_list.append(selected_epoch.item())
        self.train_accs.append(best_train_acc);
        self.valid_accs.append(best_val_acc);
        self.test_accs.append(best_test_acc)

        if fold + 1 != self.args.folds:
            self.running_train_accs = []
            self.running_valid_accs = []
            self.running_test_accs = []

            self.cnt = 0
            self.best_val = 0

    def summary(self):
        train_acc_mean = torch.tensor(self.train_accs).mean().item()
        train_acc_std = torch.tensor(self.train_accs).std().item()
        val_acc_mean = torch.tensor(self.valid_accs).mean().item()
        val_acc_std = torch.tensor(self.valid_accs).std().item()
        test_acc_mean = torch.tensor(self.test_accs).mean().item()
        test_acc_std = torch.tensor(self.test_accs).std().item()
        print('train accuracy / mean(std): {0:.5f}({1:.5f})'.format(train_acc_mean, train_acc_std))
        print('val accuracy / mean(std): {0:.5f}({1:.5f})'.format(val_acc_mean, val_acc_std))
        print('test accuracy / mean(std): {0:.5f}({1:.5f})'.format(test_acc_mean, test_acc_std))

















