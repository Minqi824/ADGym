import os
import sys
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from networks import MLP, MLP_pair, AE
import rtdl

from imblearn.over_sampling import SMOTE
from tabgan.sampler import GANGenerator

from sklearn.preprocessing import MinMaxScaler, Normalizer
from torch.utils.data import Subset, DataLoader, TensorDataset
from utils import Utils
import torch.nn.functional as F

# TODO
# using the validation (not training) set to select the best components for comparison

# we decouple the network components from the existing literature
class Components():
    def __init__(self,
                 seed: int = None,
                 data = None,
                 augmentation: str = None,
                 gan_specific: bool = False,
                 preprocess: str = None,
                 network_architecture: str = None,
                 hidden_size_list: list = None,
                 layers: int = None,
                 act_fun: str = None,
                 dropout: float = None,
                 network_initialization: str = None,
                 training_strategy = None,
                 loss_name: str = None,
                 optimizer_name: str = None,
                 batch_resample: bool = None,
                 epochs: int = None,
                 batch_size: int = None,
                 lr: float = None,
                 weight_decay: float = None):
        '''
        combination pipeline: data augmentation —— data processing —— network architecture —— network training

        **data augmentation**
        :param augmentation: data augmentation methods
        :param gan_specific: whether to use GAN for data augmentation

        **data data processing**
        :param preprocess: data preprocessing methods

        **network architecture**
        :param network_architecture: neural network architectures
        :param hidden_size_list: number of neurons in the hidden size
        :param act_fun: activation function (layer) in neural network
        :param dropout: dropout rate in neural network

        **network training**
        :param network_initialization: initialization methods of network weights
        :param training_strategy: training strategy of neural network
        :param loss_name: loss function name used for training model
        :param optimizer_name: optimizer name
        :param batch_resample: whether to use the batch resampling strategy in model training
        :param epochs: number of training epochs
        :param batch_size: training batch size
        :param lr: learning rate
        :param weight_decay: weight decay specified in the optimizer
        '''

        self.utils = Utils()
        self.seed = seed
        self.data = data

        # whether to use the gpu device (e.g., gpu for the FTTransformer network architecture)
        self.device = self.utils.get_device()

        ## data augmentation ##
        self.augmentation = augmentation
        self.gan_specific = gan_specific

        ## data preprocessing ##
        self.preprocess = preprocess

        ## network architecture ##
        self.network_architecture = network_architecture
        self.hidden_size_list = hidden_size_list
        self.layers = layers
        self.act_fun = act_fun
        self.dropout = dropout
        self.network_initialization = network_initialization

        ## network training ##
        self.training_strategy = training_strategy
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.batch_resample = batch_resample
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def gym(self, mode='small'):
        # small or large search space
        if mode == 'large':
            gyms = {}
            ## data ##
            gyms['augmentation'] = ['GAN'] if self.gan_specific else [None, 'Oversampling', 'SMOTE', 'Mixup']
            gyms['preprocess'] = ['minmax', 'normalize']

            ## network architecture ##
            gyms['network_architecture'] = ['MLP', 'AE', 'ResNet', 'FTT']
            gyms['hidden_size_list'] = [[20], [100, 20], [100, 50, 20]]
            gyms['act_fun'] = ['Tanh', 'ReLU', 'LeakyReLU']
            gyms['dropout'] = [0.0, 0.1, 0.2]
            # gyms['network_initialization'] = ['default', 'xavier_uniform', 'xavier_normal',
            #                                   'kaiming_uniform', 'kaiming_normal']
            gyms['network_initialization'] = ['default', 'xavier_normal', 'kaiming_normal']

            ## network training ##
            gyms['training_strategy'] = [None]
            gyms['loss_name'] = ['bce', 'focal', 'minus', 'inverse', 'hinge', 'deviation'] # ordinal
            gyms['optimizer_name'] = ['SGD', 'Adam', 'RMSprop']
            gyms['batch_resample'] = [True, False]
            gyms['epochs'] = [20, 50, 100]
            gyms['batch_size'] = [16, 64, 256]
            gyms['lr'] = [1e-2, 1e-3]
            gyms['weight_decay'] = [1e-2, 1e-4]

        elif mode == 'small': # we only discuss the core components in the small grid mode
            gyms = {}
            ## data ##
            gyms['augmentation'] = ['GAN'] if self.gan_specific else [None, 'Oversampling', 'SMOTE', 'Mixup']
            gyms['preprocess'] = ['minmax']

            ## network architecture ##
            gyms['network_architecture'] = ['MLP', 'AE', 'ResNet', 'FTT']
            gyms['hidden_size_list'] = [[100, 20]]
            gyms['act_fun'] = ['Tanh', 'ReLU', 'LeakyReLU']
            gyms['dropout'] = [0.0]
            gyms['network_initialization'] = ['default']

            ## network training ##
            gyms['training_strategy'] = [None]
            gyms['loss_name'] = ['bce', 'focal', 'minus', 'inverse', 'hinge', 'deviation'] # ordinal
            gyms['optimizer_name'] = ['SGD', 'Adam', 'RMSprop']
            gyms['batch_resample'] = [True, False]
            gyms['epochs'] = [50]
            gyms['batch_size'] = [256]
            gyms['lr'] = [1e-2, 1e-3]
            gyms['weight_decay'] = [1e-2]

        else:
            raise NotImplementedError

        return gyms

    # data augmentation should only perform on the training set
    def f_augmentation(self):
        if self.augmentation is None:
            pass

        elif self.augmentation == 'Oversampling':
            idx_n = np.where(self.data['y_train']==0)[0]
            idx_a = np.where(self.data['y_train']==1)[0]

            if len(idx_a) < len(idx_n):
                # resampling
                idx_a = np.random.choice(idx_a, len(idx_n), replace=True)
                idx = np.append(idx_n, idx_a)
                random.shuffle(idx)

                self.data['X_train'] = self.data['X_train'][idx]
                self.data['y_train'] = self.data['y_train'][idx]
            else:
                pass

        elif self.augmentation == 'SMOTE':
            new_X, new_y = SMOTE(random_state=self.seed).fit_resample(self.data['X_train'],
                                                                      self.data['y_train'])

            self.data['X_train'] = new_X
            self.data['y_train'] = new_y

        elif self.augmentation == 'Mixup': # mixup method need to modify the loss functions
            # https://arxiv.org/pdf/1710.09412.pdf
            # https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py

            # since mixup y would generate continuous training targets, which should therefore modify the loss function
            # we only mixup the samples belonging to the same class (mainly for the abnormal class that is the minority)
            idx_n = np.where(self.data['y_train']==0)[0]
            idx_a = np.where(self.data['y_train']==1)[0]

            if len(idx_a) < len(idx_n):
                n_augmentation = len(idx_n) - len(idx_a)
                x_augmentation = []
                for i in range(n_augmentation):
                    lam = np.random.beta(1.0, 1.0) # generate weights
                    x_augmentation.append(lam * self.data['X_train'][np.random.choice(idx_a, 1)] +
                                          (1 - lam) * self.data['X_train'][np.random.choice(idx_a, 1)])
                x_augmentation = np.vstack(x_augmentation)

                new_X = np.concatenate((self.data['X_train'], x_augmentation), axis=0)
                new_y = np.append(self.data['y_train'], np.repeat(1, n_augmentation))
                new_X, new_y = self.utils.shuffle(new_X, new_y)

                self.data['X_train'] = new_X
                self.data['y_train'] = new_y
            else:
                pass

        elif self.augmentation == 'GAN':
            # could raise error for higher version of sklearn (e.g., >=1.0)
            # we modify the GAN's params for accelerating,
            # where the original gan_params = {"batch_size": 500, "patience": 25, "epochs" : 500,}
            new_X, new_y = GANGenerator(gen_x_times=0.2, gan_params={"batch_size": 100,
                                                                     "patience": 5,
                                                                     "epochs" : 100,}).generate_data_pipe(pd.DataFrame(self.data['X_train']),
                                                                                                          pd.DataFrame(self.data['y_train'], columns=['target']),
                                                                                                          pd.DataFrame(self.data['X_train']))
            self.data['X_train'] = new_X.values
            self.data['y_train'] = new_y.values

        else:
            raise NotImplementedError

        return self

    def f_preprocess(self):
        if self.preprocess == 'minmax':
            scaler = MinMaxScaler().fit(self.data['X_train'])
        elif self.preprocess == 'normalize':
            scaler = Normalizer().fit(self.data['X_train'])
        else:
            raise NotImplementedError

        self.data['X_train'] = scaler.transform(self.data['X_train'])
        self.data['X_test'] = scaler.transform(self.data['X_test'])

        # train loader
        if self.batch_resample:
            if self.loss_name == 'ordinal':
                self.train_loader = self.utils.sampler_pairs(X_train_tensor=torch.from_numpy(self.data['X_train']).float(),
                                                             y_train=self.data['y_train'], batch_size=self.batch_size)

            else:
                X_train_resample, y_train_resample = self.utils.sampler(self.data['X_train'], self.data['y_train'], self.batch_size)
                self.train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_resample).float(),
                                               torch.tensor(y_train_resample).float()),
                                               batch_size=self.batch_size, shuffle=False, drop_last=True)
        else:
            self.train_loader = DataLoader(TensorDataset(torch.from_numpy(self.data['X_train']).float(),
                                           torch.tensor(self.data['y_train']).float()),
                                           batch_size=self.batch_size, shuffle=True, drop_last=True)

        # training tensor
        self.train_tensor = torch.from_numpy(self.data['X_train']).float()

        # testing tensor
        self.test_tensor = torch.from_numpy(self.data['X_test']).float()

        return self

    def f_init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.network_initialization == 'default':
                pass
            elif self.network_initialization == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif self.network_initialization == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif self.network_initialization == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif self.network_initialization == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            else:
                raise NotImplementedError

    def f_network(self):
        '''
        We including several network architectures that are widely used in either AD or classifiaction problem, including:
        - MLP
        - AutoEncoder
        - ResNet
        - FTTransformer
        '''
        input_size = self.data['X_train'].shape[1]

        if self.act_fun == 'Tanh':
            act = nn.Tanh()
        elif self.act_fun == 'ReLU':
            act = nn.ReLU()
        elif self.act_fun == 'LeakyReLU':
            act = nn.LeakyReLU()

        if self.network_architecture == 'MLP':
            if self.loss_name == 'ordinal':
                self.model = MLP_pair(layers=len(self.hidden_size_list), input_size=input_size, hidden_size_list=self.hidden_size_list, act_fun=act, p=self.dropout)
            else:
                self.model = MLP(layers=len(self.hidden_size_list), input_size=input_size, hidden_size_list=self.hidden_size_list, act_fun=act, p=self.dropout)

        elif self.network_architecture == 'AE':
            self.model = AE(layers=len(self.hidden_size_list), input_size=input_size, hidden_size_list=self.hidden_size_list, act_fun=act, p=self.dropout)

        # todo
        elif self.network_architecture == 'ResNet':
            # dropout_first – the dropout rate of the first dropout layer in each Block.
            # dropout_second – the dropout rate of the second dropout layer in each Block.
            # assert len(set(self.hidden_size_list)) == 1

            self.model = rtdl.ResNet.make_baseline(
                        d_in=input_size,
                        d_main=128,
                        d_hidden=self.hidden_size_list[-1],
                        dropout_first=self.dropout,
                        dropout_second=0.0,
                        n_blocks=len(self.hidden_size_list),
                        d_out=1)

        elif self.network_architecture == 'FTT':
            self.model = rtdl.FTTransformer.make_baseline(
                n_num_features=input_size,
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                n_blocks=len(self.hidden_size_list),
                ffn_d_hidden=self.hidden_size_list[-1],
                ffn_dropout=self.dropout,
                d_token=8,
                attention_dropout=0.2,
                residual_dropout=0.0,
                d_out=1)

        else:
            raise NotImplementedError

        self.model.to(self.device) # to device

        return self

    def f_training_strategy(self):
        # TODO
        pass

    def f_loss(self, s, y):
        '''
        We including several loss functions in the existing AD algorithms, including:
        - BCE (Binary Cross Entropy) loss
        - Focal loss (From the paper "Focal Loss for Dense Object Detection")
        - Minus loss (From the paper "Lifelong anomaly detection through unlearning")
        - Inverse loss (From the paper "Deep semi-supervised anomaly detection")
        - Hinge loss (From the paper "Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection")
        - Deviation loss (From the paper "Deep anomaly detection with deviation networks")
        - Ordinal loss (to do) (From the paper "Deep Weakly-supervised Anomaly Detection")
        '''
        ranking_loss = torch.nn.MarginRankingLoss(margin=5.0) # for hinge loss

        s = s.squeeze()
        s_n = s[y == 0] # anomaly score of normal (unlabeled) samples
        s_a = s[y == 1] # anomaly score of labeled anomalies

        if self.loss_name == 'bce':
            loss = F.binary_cross_entropy_with_logits(input=s, target=y, reduction="mean")

        elif self.loss_name == 'focal':
            loss = self.utils.sigmoid_focal_loss(inputs=s, targets=y, reduction="mean")

        elif self.loss_name == 'minus':
            loss = torch.mean(s_n + torch.max(torch.zeros_like(s_a), 5.0 - s_a))

        elif self.loss_name == 'inverse':
            loss = torch.mean(torch.pow(s_n, torch.ones_like(s_n))) + torch.mean(torch.pow(s_a, -1 * torch.ones_like(s_a)))

        elif self.loss_name == 'hinge':
            loss = ranking_loss(s_a, s_n, torch.ones_like(s_a))

        elif self.loss_name == 'deviation':
            ref = torch.randn(5000)  # sampling references from the normal distribution
            s_n = (s_n - torch.mean(ref)) / torch.std(ref) # normalized anomaly score of normal samples
            s_a = (s_a - torch.mean(ref)) / torch.std(ref) # normalized anomaly score of labeled anomalies

            inlier_loss = torch.abs(s_n)
            outlier_loss = torch.max(torch.zeros_like(s_a), 5.0 - s_a)
            loss = torch.mean(inlier_loss + outlier_loss)

        elif self.loss_name == 'ordinal':
            loss = torch.mean(torch.abs(y - s))

        else:
            raise NotImplementedError

        return loss

    def f_optimizer(self):
        # TODO: weight decay
        if self.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        elif self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        elif self.optimizer_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        else:
            raise NotImplementedError

        return self

    def f_train(self):
        self.utils.set_seed(self.seed)

        # data augmentation
        self.f_augmentation()

        # data preprocessing
        self.f_preprocess()

        # network initialization
        self.f_network() # build network
        self.model.apply(self.f_init_weights) # network weight initialization

        # optimizer
        self.f_optimizer()

        # fitting
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                # data
                X, y = batch
                # to device
                if self.loss_name == 'ordinal':
                    X_left, X_right = X
                    X_left = X_left.to(self.device); X_right = X_right.to(self.device); y = y.to(self.device)
                else:
                    X = X.to(self.device); y = y.to(self.device)

                # clear gradient
                self.model.zero_grad()

                # loss forward
                if self.network_architecture == 'FTT':
                    s = self.model(x_num=X, x_cat=None)
                elif self.loss_name == 'ordinal':
                    s = self.model(X_left=X_left, X_right=X_right)
                else:
                    s = self.model(X)

                loss = self.f_loss(s, y)

                # loss backward
                loss.backward()

                # gradient update
                self.optimizer.step()

        return self

    @torch.no_grad()
    def f_predict_score(self, num=30):
        self.model.eval()

        if self.network_architecture == 'FTT':
            score_train = self.model(self.train_tensor.to(self.device), x_cat=None)
            score_train = score_train.squeeze().cpu().numpy()

            score_test = self.model(self.test_tensor.to(self.device), x_cat=None)
            score_test = score_test.squeeze().cpu().numpy()

        elif self.loss_name == 'ordinal':
            def f_score(X_test):
                score_test = []
                X_train = self.train_tensor.to(self.device)

                for i in range(X_test.size(0)):
                    # postive and negative sample indices in the training set
                    index_a = np.random.choice(np.where(self.data['y_train'] == 1)[0], num, replace=True)
                    index_u = np.random.choice(np.where(self.data['y_train'] == 0)[0], num, replace=True)
                    X_train_a_tensor = X_train[index_a]
                    X_train_u_tensor = X_train[index_u]

                    score_a_x = self.model(X_train_a_tensor, torch.cat(num * [X_test[i].view(1, -1)]))
                    score_x_u = self.model(torch.cat(num * [X_test[i].view(1, -1)]), X_train_u_tensor)
                    score_sub = torch.mean(score_a_x + score_x_u)
                    score_test.append(score_sub.cpu().item())

                score_test = np.array(score_test)
                return score_test

            score_train = f_score(X_test=self.train_tensor.to(self.device))
            score_test = f_score(X_test=self.test_tensor.to(self.device))

        else:
            score_train = self.model(self.train_tensor.to(self.device))
            score_train = score_train.squeeze().cpu().numpy()

            score_test = self.model(self.test_tensor.to(self.device))
            score_test = score_test.squeeze().cpu().numpy()

        metrics_train = self.utils.metric(y_true=self.data['y_train'], y_score=score_train, pos_label=1)
        metrics_test = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

        return (score_train, score_test), (metrics_train, metrics_test)
