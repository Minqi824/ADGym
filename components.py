import numpy as np
import torch
from torch import nn
from networks import MLP, AE
import rtdl
import argparse

from sklearn.preprocessing import MinMaxScaler, Normalizer
from torch.utils.data import Subset, DataLoader, TensorDataset
from utils import Utils


# TODO
# data augmentation

# we decouple the network components from the existing literature
class Components():
    def __init__(self,
                 data=None,
                 augmentation:str=None,
                 preprocess:str=None,
                 network_architecture:str=None,
                 layers:int=None,
                 hidden_size_list:list=None,
                 act_fun:str=None,
                 dropout:float=None,
                 training_strategy=None,
                 loss_name:str=None,
                 optimizer_name:str=None,
                 batch_resample:bool=None,
                 epochs:int=None,
                 batch_size:int=None,
                 lr:float=None,
                 weight_decay:float=None):

        self.utils = Utils()
        self.data = data

        # whether to use the gpu device
        if network_architecture == 'FTT':
            self.device = self.utils.get_device(gpu_specific=True)
        else:
            self.device = self.utils.get_device(gpu_specific=False)

        ## data ##
        self.augmentation = augmentation
        self.preprocess = preprocess

        ## network architecture ##
        self.network_architecture = network_architecture
        self.layers = layers
        self.hidden_size_list = hidden_size_list
        self.act_fun = act_fun
        self.dropout = dropout

        ## network fitting ##
        self.training_strategy = training_strategy
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.batch_resample = batch_resample
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def gym(self):
        gyms = {}

        ## data ##
        gyms['augmentation'] = [None]
        gyms['preprocess'] = ['minmax', 'normalize']

        ## network architecture ##
        gyms['network_architecture'] = ['MLP', 'AE', 'ResNet', 'FTT']
        gyms['layers'] = [1, 2, 3]
        gyms['hidden_size_list'] = [[20], [100, 20], [100, 50, 20]]
        gyms['act_fun'] = ['Tanh', 'ReLU', 'LeakyReLU']
        gyms['dropout'] = [0.0, 0.1, 0.3]

        ## network fitting ##
        gyms['training_strategy'] = [None]
        gyms['loss_name'] = ['minus', 'inverse', 'hinge', 'deviation']
        gyms['optimizer_name'] = ['SGD', 'Adam', 'RMSprop']
        gyms['batch_resample'] = [True]
        gyms['epochs'] = [20, 50, 100]
        gyms['batch_size'] = [16, 64, 256]
        gyms['lr'] = [1e-2, 1e-3]
        gyms['weight_decay'] = [1e-2, 1e-4]

        return gyms

    def f_augmentation(self):
        # TODO
        pass

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
            X_train_resample, y_train_resample = self.utils.sampler(self.data['X_train'], self.data['y_train'], self.batch_size)
            train_tensor = TensorDataset(torch.from_numpy(X_train_resample).float(),
                                         torch.tensor(y_train_resample).float())
            self.train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)
        else:
            train_tensor = TensorDataset(torch.from_numpy(self.data['X_train']).float(),
                                         torch.tensor(self.data['y_train']).float())
            self.train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # testing tensor
        self.test_tensor = torch.from_numpy(self.data['X_test']).float()

        return self

    def f_network(self):
        '''
        We including several network architectures, including:
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
            self.model = MLP(layers=self.layers, input_size=input_size, hidden_size_list=self.hidden_size_list, act_fun=act, p=self.dropout)

        elif self.network_architecture == 'AE':
            self.model = AE(layers=self.layers, input_size=input_size, hidden_size_list=self.hidden_size_list, act_fun=act, p=self.dropout)

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
                        n_blocks=self.layers,
                        d_out=1)

        elif self.network_architecture == 'FTT':
            self.model = rtdl.FTTransformer.make_default(
                n_num_features=input_size,
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                n_blocks=self.layers,
                d_out=1,
            )

        else:
            raise NotImplementedError

        self.model.to(self.device)

        return self

    def f_training_strategy(self):
        # TODO
        pass

    def f_loss(self, s_n, s_a):
        '''
        We including several loss functions in the existing AD algorithms, including:
        - Minus loss
        - Inverse loss
        - Hinge loss
        - Deviation loss
        - Ordinal loss (to do)
        '''
        ranking_loss = torch.nn.MarginRankingLoss(margin=5.0) # for hinge loss

        if self.loss_name == 'minus':
            loss = torch.mean(s_n + torch.max(torch.zeros_like(s_a), 5.0 - s_a))

        elif self.loss_name == 'inverse':
            loss = torch.mean(torch.pow(s_n, torch.ones_like(s_n)) + torch.pow(s_a, -1 * torch.ones_like(s_a)))

        elif self.loss_name == 'hinge':
            loss = ranking_loss(s_a, s_n, torch.ones_like(s_a))

        elif self.loss_name == 'deviation':
            ref = torch.randn(5000)  # sampling from the normal distribution
            s_n = (s_n - torch.mean(ref)) / torch.std(ref) # normalized anomaly score
            s_a = (s_a - torch.mean(ref)) / torch.std(ref) # normalized anomaly score

            inlier_loss = torch.abs(s_n)
            outlier_loss = torch.max(torch.zeros_like(s_a), 5.0 - s_a)
            loss = torch.mean(inlier_loss + outlier_loss)
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
        # data augmentation
        self.f_augmentation()

        # data preprocessing
        self.f_preprocess()

        # network initialization
        self.f_network()

        # optimizer
        self.f_optimizer()

        # fitting
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                # data
                X, y = batch
                # to device
                X = X.to(self.device); y = y.to(self.device)

                # clear gradient
                self.model.zero_grad()

                # loss forward
                if self.network_architecture == 'FTT':
                    s = self.model(x_num=X, x_cat=None)
                else:
                    s = self.model(X)

                s = s.squeeze()
                s_n = s[y==0]
                s_a = s[y==1]
                loss = self.f_loss(s_n, s_a)

                # loss backward
                loss.backward()

                # gradient update
                self.optimizer.step()

        return self

    @torch.no_grad()
    def f_predict_score(self):
        self.model.eval()

        if self.network_architecture == 'FTT':
            score_test = self.model(self.test_tensor.to(self.device), x_cat=None)
        else:
            score_test = self.model(self.test_tensor.to(self.device))

        score_test = score_test.squeeze().numpy()
        metrics = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

        return metrics


# X_train = np.random.randn(1000, 6)
# X_test = np.random.randn(1000, 6)
#
# y_train = np.random.choice([0,1], 1000)
# y_test = np.random.choice([0,1], 1000)
#
# data = {'X_train': X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
#
# # com = Components(data=data,
# #                  augmentation=None,
# #                  preprocess='minmax',
# #                  network_architecture='AE',
# #                  layers=2,
# #                  hidden_size_list=[100, 20],
# #                  act_fun='ReLU',
# #                  dropout=0.1,
# #                  training_strategy=None,
# #                  loss_name='deviation',
# #                  optimizer_name='SGD',
# #                  batch_resample=True,
# #                  epochs=50,
# #                  batch_size=64,
# #                  lr=1e-3,
# #                  weight_decay=1e-2)
#
#
# com = Components(data=data,augmentation= None,
#                  preprocess= 'normalize',
#                  network_architecture= 'AE',
#                  layers= 2,
#                  hidden_size_list= [100, 50, 20],
#                  act_fun= 'Tanh',
#                  dropout= 0.3,
#                  training_strategy= None,
#                  loss_name= 'deviation',
#                  optimizer_name= 'Adam',
#                  batch_resample= False,
#                  epochs= 100,
#                  batch_size= 256,
#                  lr= 0.01,
#                  weight_decay= 0.01)
#
# com.f_train()
# metrics = com.f_predict_score()
# print(metrics)
