import torch
from networks import MLP, AE
import rtdl

from sklearn.preprocessing import MinMaxScaler, Normalizer


# we decouple the network components from the existing literature
class Components():
    def __init__(self, data,
                 augmentation=None,
                 preprocess=None,
                 network=None,
                 strategy=None,
                 loss=None):

        # input data
        self.data = data

        # global parameters
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.network = network
        self.strategy = strategy
        self.loss = loss

    def data_augmentation(self):




    def data_preprocessing(self):
        if self.preprocess == 'minmax':
            scaler = MinMaxScaler().fit(self.data['X_train'])
        elif self.preprocess == 'normalize':
            scaler = Normalizer().fit(self.data['X_train'])

        self.data['X_train'] = scaler.transform(self.data['X_train'])
        self.data['X_test'] = scaler.transform(self.data['X_test'])


    def network(self):
        '''
        We including several network architectures, including:
        - MLP
        - AutoEncoder
        - ResNet
        - FTTransformer
        '''

        input_size = self.data['X_train'].shape[1]

        if self.network == 'MLP':
            self.model = MLP(input_size=input_size, act_fun=)

        elif self.network == 'AE':
            self.model = AE(input_size=input_size, act_fun=)

        elif self.network == 'ResNet':
            self.model = rtdl.ResNet.make_baseline(
                        d_in=input_size,
                        d_main=128,
                        d_hidden=256,
                        dropout_first=0.2,
                        dropout_second=0.0,
                        n_blocks=2,
                        d_out=1)
            self.model.add_module('reg', nn.BatchNorm1d(num_features=1))

        elif self.network == 'FTT':
            model = rtdl.FTTransformer.make_default(
                n_num_features=input_size,
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                d_out=1,
            )
            model.add_module('reg', nn.BatchNorm1d(num_features=1))

        else:
            raise NotImplementedError

    def training_strategy(self):
        pass

    def loss(self, s_n, s_a):
        '''
        We including several loss functions in the existing AD algorithms, including:
        - Minus loss
        - Inverse loss
        - Hinge loss
        - Deviation loss
        - Ordinal loss (to do)
        '''
        ranking_loss = torch.nn.MarginRankingLoss(margin=5.0) # for hinge loss

        if self.loss == 'minus':
            loss = torch.mean(s_n + torch.max(torch.zeros_like(s_a), 5.0 - s_a))
        elif self.loss == 'inverse':
            loss = torch.mean(torch.pow(s_n, torch.ones_like(s_n)) + torch.pow(s_a, -1 * torch.ones_like(s_a)))
        elif self.loss == 'hinge':
            loss = ranking_loss(s_a, s_n, torch.ones_like(s_a))
        elif self.loss == 'deviation':
            ref = torch.randn(5000)  # sampling from the normal distribution
            s_n = (s_n - torch.mean(ref)) / torch.std(ref) # normalized anomaly score
            s_a = (s_a - torch.mean(ref)) / torch.std(ref) # normalized anomaly score

            inlier_loss = torch.abs(s_n)
            outlier_loss = torch.max(torch.zeros_like(s_a), 5.0 - s_a)
            loss = torch.mean(inlier_loss + outlier_loss)
        else:
            raise NotImplementedError

        return loss



    def fit(self):
        return model

    def predict_score(self):

        return metrics
