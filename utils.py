import numpy as np
import random
import torch
# metric
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
from torch import nn

class Utils():
    def __init__(self):
        pass

    # remove randomness
    def set_seed(self, seed):
        # basic seed
        np.random.seed(seed)
        random.seed(seed)

        # pytorch seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # generate unique value
    def unique(self, a, b):
        u = 0.5 * (a + b) * (a + b + 1) + b
        return int(u)

    def get_device(self, gpu_specific=True):
        if gpu_specific:
            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                print(f'number of gpu: {n_gpu}')
                print(f'cuda name: {torch.cuda.get_device_name(0)}')
                print('GPU is on')
            else:
                print('GPU is off')

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device

    def criterion(self, y_true, y_pred, mode=None):
        assert torch.is_tensor(y_true) and torch.is_tensor(y_pred)
        if mode == 'pearson':
            x = y_pred
            y = y_true
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            metric = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        elif mode == 'ranknet':
            n = y_pred.size(0)

            assert y_true.ndim == 1 and y_pred.ndim == 1
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

            mask = ~torch.eye(n, dtype=torch.bool)
            p_ij = torch.sign(y_true - y_true.T)
            p_ij[p_ij == -1] = 0
            s_ij = torch.sigmoid((y_pred - y_pred.T) * 100)

            p_ij = p_ij[mask].view(n, n - 1)
            s_ij = s_ij[mask].view(n, n - 1)

            metric = -F.binary_cross_entropy(s_ij, p_ij)

        elif mode == 'mse':
            criterion = nn.MSELoss()
            metric = -criterion(y_pred, y_true)

        # elif mode == 'weighted_mse':
        #     # 定义起始值、结束值和衰减因子
        #     start = 1.00
        #     end = 0.01
        #     decay_factor = 1.0
        #
        #     # 生成等间隔的向量
        #     t = torch.linspace(0, 1, y_pred.size(0))
        #     # 计算指数函数
        #     exponential_decay = torch.exp(torch.log(torch.tensor(end / start)) * decay_factor * t) * start
        #     exponential_decay = exponential_decay.to(y_pred.device)
        #
        #     idx_sort = torch.argsort(y_true)
        #     y_pred = y_pred[idx_sort]
        #     y_true = y_true[idx_sort]
        #
        #     metric = torch.sum((torch.pow((y_pred - y_true), 2) * exponential_decay))
        #     metric = -metric

        elif mode == 'weighted_mse':
            # 定义起始值、结束值和衰减因子
            start = 1.00
            end = 0.01
            decay_factor = 0.5

            # 生成等间隔的向量
            t = torch.linspace(0, 1, y_pred.size(0))
            # 计算指数函数
            exponential_decay = torch.exp(torch.log(torch.tensor(end / start)) * decay_factor * t) * start
            exponential_decay = exponential_decay.to(y_pred.device)

            idx_sort = torch.argsort(0.8 * y_true + 0.2 * y_pred)
            y_pred = y_pred[idx_sort]
            y_true = y_true[idx_sort]

            metric = torch.sum((torch.pow((y_pred - y_true), 2) * exponential_decay))
            metric = -metric

        else:
            raise NotImplementedError

        return metric

    @torch.no_grad()
    def evaluate(self, model, val_loader, device, mode=None):
        model.eval()
        y_pred, y_true, val_metric_batch = [], [], []
        for batch in val_loader:
            batch_meta_features, batch_la, batch_components, batch_y = [_.to(device) for _ in batch]
            _, pred = model(batch_meta_features, batch_la.unsqueeze(1), batch_components)

            y_pred_batch = pred.squeeze().cpu(); y_pred.extend(y_pred_batch.tolist())
            y_true_batch = batch_y.squeeze().cpu(); y_true.extend(y_true_batch.tolist())
            val_metric_batch.append(self.criterion(y_true=y_true_batch, y_pred=y_pred_batch, mode=mode))

        if mode == 'ranknet':
            val_metric = np.mean(val_metric_batch)
        else:
            val_metric = self.criterion(y_true=torch.tensor(y_true), y_pred=torch.tensor(y_pred), mode=mode)

        return val_metric

    @torch.no_grad()
    def evaluate_end2end(self, model, meta_data_val, device, mode=None):
        model.eval()
        y_pred, y_true, val_metric_batch = [], [], []
        for meta_data_batch in meta_data_val:
            X_list, y_list, la_list, components, targets = meta_data_batch
            _, _, pred = model(X_list, y_list, la_list, components)

            y_pred_batch = pred.squeeze().cpu(); y_pred.extend(y_pred_batch.tolist())
            y_true_batch = targets.squeeze().cpu(); y_true.extend(y_true_batch.tolist())
            val_metric_batch.append(self.criterion(y_true=y_true_batch, y_pred=y_pred_batch, mode=mode))

        if mode == 'ranknet':
            val_metric = np.mean(val_metric_batch)
        else:
            val_metric = self.criterion(y_true=torch.tensor(y_true), y_pred=torch.tensor(y_pred), mode=mode)

        return val_metric

    def coral(self, Dt, Ds, epsilon=1e-6):
        Cs = np.cov(Ds, rowvar=False) + np.eye(Ds.shape[1])
        Ct = np.cov(Dt, rowvar=False) + np.eye(Dt.shape[1])

        Ct_inverse_sqrt = np.power(np.linalg.inv(Ct), 0.5)
        Ct_inverse_sqrt[np.isnan(Ct_inverse_sqrt)] = epsilon
        Cs_sqrt = np.power(Cs, 0.5)
        Cs_sqrt[np.isnan(Cs_sqrt)] = epsilon

        Dt = np.dot(Dt, Ct_inverse_sqrt)
        Dt = np.dot(Dt, Cs_sqrt)

        return Dt

    # metric
    def metric(self, y_true, y_score, pos_label=1):
        aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
        aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=1)

        return {'aucroc':aucroc, 'aucpr':aucpr}

    # shuffling function
    def shuffle(self, X, y):
        idx = np.arange(len(y))
        random.shuffle(idx)

        return X[idx], y[idx]

    # resampling function
    def sampler(self, X_train, y_train, batch_size):
        index_u = np.where(y_train == 0)[0]
        index_a = np.where(y_train == 1)[0]

        n = 0
        while len(index_u) >= batch_size:
            self.set_seed(n)
            index_u_batch = np.random.choice(index_u, batch_size // 2, replace=False)
            index_u = np.setdiff1d(index_u, index_u_batch)

            index_a_batch = np.random.choice(index_a, batch_size // 2, replace=True)

            # batch index
            index_batch = np.append(index_u_batch, index_a_batch)
            # shuffle
            np.random.shuffle(index_batch)

            if n == 0:
                X_train_new = X_train[index_batch]
                y_train_new = y_train[index_batch]
            else:
                X_train_new = np.append(X_train_new, X_train[index_batch], axis=0)
                y_train_new = np.append(y_train_new, y_train[index_batch])
            n += 1

        return X_train_new, y_train_new

    def sampler_pairs(self, X_train_tensor, y_train, batch_size, batch_num=20, s_a_a=8.0, s_a_u=4.0, s_u_u=0.0):
        '''
        X_train_tensor: the input X in the torch.tensor form
        y_train: label in the numpy.array form
        batch_num: generate how many batches in one epoch
        batch_size: the batch size
        '''
        data_loader = []
        index_a = np.where(y_train == 1)[0]
        index_u = np.where(y_train == 0)[0]

        for i in range(batch_num):  # i.e., drop_last = True
            index = []
            # 分别是(a,a); (a,u); (u,u)共6部分样本
            for j in range(6):
                if j < 3:
                    index_sub = np.random.choice(index_a, batch_size // 4, replace=True)
                    index.append(list(index_sub))

                if j == 3:
                    index_sub = np.random.choice(index_u, batch_size // 4, replace=True)  # unlabel部分可以变为False
                    index.append(list(index_sub))

                if j > 3:
                    index_sub = np.random.choice(index_u, batch_size // 2, replace=True)  # unlabel部分可以变为False
                    index.append(list(index_sub))

            # index[0] + index[1] = (a,a), batch / 4
            # index[2] + index[2] = (a,u), batch / 4
            # index[4] + index[5] = (u,u), batch / 2
            index_left = index[0] + index[2] + index[4]
            index_right = index[1] + index[3] + index[5]

            X_train_tensor_left = X_train_tensor[index_left]
            X_train_tensor_right = X_train_tensor[index_right]

            # generate label
            y_train_new = np.append(np.repeat(s_a_a, batch_size // 4), np.repeat(s_a_u, batch_size // 4))
            y_train_new = np.append(y_train_new, np.repeat(s_u_u, batch_size // 2))
            y_train_new = torch.from_numpy(y_train_new).float()

            # shuffle
            index_shuffle = np.arange(len(y_train_new))
            random.shuffle(index_shuffle)

            X_train_tensor_left = X_train_tensor_left[index_shuffle]
            X_train_tensor_right = X_train_tensor_right[index_shuffle]
            y_train_new = y_train_new[index_shuffle]

            # save, 注意left和right顺序
            data_loader.append([[X_train_tensor_left, X_train_tensor_right], y_train_new])

        return data_loader

    def sigmoid_focal_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "none",
    ):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #     _log_api_usage_once(sigmoid_focal_loss)
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss