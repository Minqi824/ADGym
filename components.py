import torch
from torch import nn
from networks import MLP, AE
import rtdl

from sklearn.preprocessing import MinMaxScaler, Normalizer
from torch.utils.data import Subset, DataLoader, TensorDataset
from utils import Utils


# we decouple the network components from the existing literature
class Components():
    def __init__(self, data,
                 augmentation=None,
                 preprocess=None,
                 network_name=None,
                 training_strategy=None,
                 loss_name=None,
                 optimizer_name=None):

        self.utils = Utils()

        # input data
        self.data = data

        # global parameters
        self.augmentation = augmentation # data augmentation
        self.preprocess = preprocess # data preprocessing
        self.network_name = network_name # network architecture name
        self.training_strategy = training_strategy # training strategy
        self.loss_name = loss_name # loss function name
        self.optimizer_name = optimizer_name # optmizer name

        self.batch_resample = True
        self.epochs = 50
        self.batch_size = 64
        self.lr = 1e-3

    def gym(self): #
        pool = {}

        pool['augmentation'] = ['None']
        pool['preprocess'] = ['minmax', 'normalize']
        pool['network_name'] = ['MLP', 'AE', 'ResNet', 'FTT']
        pool['training_strategy'] = ['None']
        pool['loss_name'] = ['minus', 'inverse', 'hinge', 'deviation']
        pool['optimizer_name'] = ['SGD', 'Adam', 'RMSprop']

        return pool

    def f_augmentation(self):
        # TODO
        pass

    def f_preprocess(self):
        if self.preprocess == 'minmax':
            scaler = MinMaxScaler().fit(self.data['X_train'])
        elif self.preprocess == 'normalize':
            scaler = Normalizer().fit(self.data['X_train'])

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

        if self.network_name == 'MLP':
            self.model = MLP(input_size=input_size, act_fun=nn.ReLU())

        elif self.network_name == 'AE':
            self.model = AE(input_size=input_size, act_fun=nn.ReLU())

        elif self.network_name == 'ResNet':
            self.model = rtdl.ResNet.make_baseline(
                        d_in=input_size,
                        d_main=128,
                        d_hidden=256,
                        dropout_first=0.2,
                        dropout_second=0.0,
                        n_blocks=2,
                        d_out=1)
            self.model.add_module('reg', nn.BatchNorm1d(num_features=1))

        elif self.network_name == 'FTT':
            self.model = rtdl.FTTransformer.make_default(
                n_num_features=input_size,
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                d_out=1,
            )
            self.model.add_module('reg', nn.BatchNorm1d(num_features=1))

        else:
            raise NotImplementedError

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
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)

        else:
            raise NotImplementedError

        return self

    def train(self):
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

                # clear gradient
                self.model.zero_grad()

                # loss forward
                _, s = self.model(X)
                s_n = s[y==0]
                s_a = s[y==1]
                loss = self.f_loss(s_n, s_a)

                # loss backward
                loss.backward()

                # gradient update
                self.optimizer.step()

        return self

    @torch.no_grad()
    def predict_score(self):
        self.model.eval()

        _, score_test = self.model(self.test_tensor)
        score_test = score_test.numpy()
        metrics = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

        return metrics
