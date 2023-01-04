import numpy as np
import random
import torch
# metric
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F

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

    def get_device(self, gpu_specific=False):
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