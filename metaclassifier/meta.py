# Training a meta classifier to predict detection performance by given:
# 1. meta-feature
# meta-feature can be extracted by using either end2end method like dataset2vec,
# or some two-stage method like the meta-feature extracted in the MetaOD method
# meta-feature is mainly used for transferring knowledge across different datasets (i.e., transferring knowledge in X)

# 2. number of labeled anomalies
# because we at least know how many labeled anomalies exist in the testing dataset,
# therefore the number of labeled anomalies can be served as an important indicator in the meta-classifier,
# since model performance is highly correlated with the number of labeled anomalies, e.g.,
# some loss functions like deviation loss or simple network architectures like MLP are more competitive on few anomalies,
# while more complex network architectures like Transformer are more efficient when more labeled anomalies are available.

# 3. pipeline components
# instead of one-hot encoding, we encode the pipeline components to continuous embedding,
# therefore realizing end-to-end learning of the component representations
# the learned component representations can be used for visualization,
# e.g., similar components can achieve similar detection performances

# ToDo
# end-to-end meta-feature (should we follow a pretrained-finetune process?)

import time
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn import preprocessing
from torch import nn
import torch
from torch.utils.data import Subset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_generator import DataGenerator
from utils import Utils
from metaclassifier.networks import meta_predictor, meta_predictor_end2end
from metaclassifier.fit import fit, fit_end2end

class meta():
    def __init__(self,
                 seed:int = 42,
                 metric: str='AUCPR',
                 suffix: str='',
                 grid_mode: int='small',
                 grid_size: int=10000,
                 gan_specific: bool=False,
                 test_dataset: str=None,
                 test_la: int=None):

        self.seed = seed
        self.metric = metric
        self.suffix = suffix
        self.grid_mode = grid_mode
        self.grid_size = grid_size
        self.gan_specific = gan_specific

        self.test_dataset = test_dataset
        self.test_la = test_la

        self.utils = Utils()
        self.data_generator = DataGenerator()

    def components_process(self, result):
        assert isinstance(result, pd.DataFrame)

        # we only compare the differences in diverse components
        components_list = [ast.literal_eval(_) for _ in result['Components']] # list of dict
        keys = list(ast.literal_eval(result['Components'][0]).keys())
        keys_diff = []

        for k in keys:
            options = [str(_[k]) for _ in components_list if _[k] is not None]
            # delete components that only have unique or None value
            if len(set(options)) == 1 or len(options) == 0:
                continue
            else:
                keys_diff.append(k)

        # save components as dataframe
        components_list_diff = []
        for c in components_list:
            components_list_diff.append({k: c[k] for k in keys_diff})

        components_df = pd.DataFrame(components_list_diff)
        components_df = components_df.replace([None], 'None')

        # encode components to int index for preparation
        components_df_index = components_df.copy()
        for col in components_df_index.columns:
            components_df_index[col] = preprocessing.LabelEncoder().fit_transform(components_df_index[col])

        return components_df_index

    # TODO
    # two versions of meta classifer: using MetaOD feature extractor and end2end learning
    # add some constraints to improve the training process of meta classifier, e.g.,
    # we can remove some unimportant components after detailed analysis
    def meta_fit2test(self):
        # set seed for reproductive results
        self.utils.set_seed(self.seed)
        # generate training data for meta predictor
        meta_features, las, components, performances = [], [], [], []

        for la in [5, 10, 25, 50]:
            result = pd.read_csv('../result/result-' + self.metric + '-'.join([self.suffix, str(la), self.grid_mode, str(self.grid_size), 'GAN', str(self.gan_specific)]) + '.csv')
            result.rename(columns={'Unnamed: 0': 'Components'}, inplace=True)

            # remove dataset of testing task
            result.drop([self.test_dataset], axis=1, inplace=True)
            assert self.test_dataset not in result.columns

            # transform result dataframe for preparation
            components_df_index = self.components_process(result)

            for i in range(result.shape[0]):
                for j in range(1, result.shape[1]):
                    if not pd.isnull(result.iloc[i, j]) and result.columns[j] != self.test_dataset:  # set nan to 0?
                        meta_feature = np.load('../datasets/meta-features/' + 'meta-features-' + result.columns[j] + '-' + str(la) + '-1.npz', allow_pickle=True)

                        # preparing training data for meta classifier
                        # note that we only extract meta features in training set of both training & testing tasks
                        meta_features.append(meta_feature['data'])
                        las.append(la)
                        components.append(components_df_index.iloc[i, :].values)
                        performances.append(result.iloc[i, j])

        del la

        meta_features = np.stack(meta_features)
        components = np.stack(components)

        # fillna in extracted meta-features
        meta_features = pd.DataFrame(meta_features).fillna(0).values
        # min-max scaling for meta-features
        scaler_meta_features = MinMaxScaler().fit(meta_features)
        meta_features = scaler_meta_features.transform(meta_features)

        # to tensor
        meta_features = torch.from_numpy(meta_features).float()
        las = torch.tensor(las).float()
        components = torch.from_numpy(components).float()
        performances = torch.tensor(performances).float()
        # to dataloader
        train_loader = DataLoader(TensorDataset(meta_features, las, components, performances),
                                  batch_size=512, shuffle=True, drop_last=True)

        # initialization meta classifier
        model = meta_predictor(n_col=components.size(1),
                               n_per_col=[max(components[:, i]).item() + 1 for i in range(components.size(1))])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # fitting meta classifier
        print('fitting meta classifier...')
        fit(train_loader, model, optimizer)

        # testing
        # 1. meta-feature for testing dataset
        meta_feature_test = np.load('../datasets/meta-features/' + 'meta-features-' + self.test_dataset + '-' + str(self.test_la) + '-1.npz', allow_pickle=True)
        meta_feature_test = meta_feature_test['data']
        meta_feature_test = np.stack([meta_feature_test for i in range(components_df_index.shape[0])])
        meta_feature_test = pd.DataFrame(meta_feature_test).fillna(0).values
        meta_feature_test = scaler_meta_features.transform(meta_feature_test)
        meta_feature_test = torch.from_numpy(meta_feature_test).float()

        # 2. number of labeled anomalies in testing dataset
        la_test = np.repeat(self.test_la, components_df_index.shape[0])
        la_test = torch.tensor(la_test).float()

        # 3. components (predefined)
        components_test = torch.from_numpy(components_df_index.values).float()

        # predicting
        with torch.no_grad():
            _, pred = model(meta_feature_test, la_test.unsqueeze(1), components_test)

        # since we have already train-test all the components on each dataset,
        # we can only inquire the experiment result with no information leakage
        result = pd.read_csv('../result/result-' + self.metric + '-'.join([self.suffix, str(self.test_la), self.grid_mode, str(self.grid_size), 'GAN', str(self.gan_specific)]) + '.csv')
        for _ in torch.argsort(-pred.squeeze()):
            pred_performance = result.loc[_.item(), self.test_dataset]
            if not pd.isnull(pred_performance):
                break

        return pred_performance

# run experiments for comparing proposed meta classifier and current SOTA methods
for metric in ['AUCPR']:
    # result of current SOTA models
    result_SOTA_semi = pd.read_csv('../result/' + metric + '-SOTA-semi-supervise.csv')
    result_SOTA_sup = pd.read_csv('../result/' + metric + '-SOTA-supervise.csv')
    result_SOTA = result_SOTA_semi.merge(result_SOTA_sup, how='inner', on='Unnamed: 0')
    del result_SOTA_semi, result_SOTA_sup

    meta_classifier_performance = np.repeat(0, result_SOTA.shape[0]).astype(float)
    for i in range(result_SOTA.shape[0]):
        test_dataset, test_la, test_seed = ast.literal_eval(result_SOTA.iloc[i, 0])
        print(f'Experiments on meta classifier: Dataset: {test_dataset}, la: {test_la}, seed: {test_seed}')

        run_meta = meta(metric=metric,
                        suffix='',
                        grid_mode='small',
                        grid_size=3000,
                        gan_specific=False,
                        test_dataset=test_dataset,
                        test_la=test_la)

        try:
            perf = run_meta.meta_fit2test()
            meta_classifier_performance[i] = perf
        except Exception as error:
            print(f'Something error when training meta-classifier: {error}')
            meta_classifier_performance[i] = -1

        result_SOTA['Meta'] = meta_classifier_performance
        result_SOTA.to_csv('../result/' + metric + '-meta.csv', index=False)



# end2end version (TODO)
# result = pd.read_csv('../result/result-AUCPR-5-small-10000-GAN-True.csv')
# result.rename(columns={'Unnamed: 0':'Components'}, inplace=True)
# # components
# components_df_index = components_process(result)
#
#
# metric = 'AUCPR'
#
# # current SOTA
# result_SOTA_semi = pd.read_csv(os.path.join('../result', metric + '_SOTA_semi-supervise.csv'))
# result_SOTA_sup = pd.read_csv(os.path.join('../result', metric + '_SOTA_supervise.csv'))
#
# assert all(result_SOTA_semi.iloc[:, 0].values == result_SOTA_sup.iloc[:, 0].values)
# result_SOTA = pd.concat([result_SOTA_semi, result_SOTA_sup.iloc[:, 1:]], axis=1)
# result_SOTA.rename(columns={'Unnamed: 0':'Components'}, inplace=True)
# del result_SOTA_semi, result_SOTA_sup
#
# pred_performances = []
# for i in tqdm(range(result_SOTA.shape[0])):
#     self.test_dataset, test_la, _ = ast.literal_eval(result_SOTA['Components'].values[i])
#
# # generate training data for meta predictor
# meta_data = []
# for la in [5, 10, 25, 50]:
#     # result = pd.read_csv('result/result_' + metric + '_' + str(la) + '_small_500.csv')
#     result = pd.read_csv('result-AUCPR-5-small-10000-GAN-True.csv')
#     result.rename(columns={'Unnamed: 0':'Components'}, inplace=True)
#     # components
#     components_df_index = components_process(result)
#
#     # meta data batch
#     for i in tqdm(range(1, result.shape[1])):
#         current_dataset = result.columns[i]
#         if current_dataset == self.test_dataset:
#             continue
#
#         # generate dataset
#         data_generator.dataset = current_dataset
#         data_generator.seed = 1
#         data = data_generator.generator(la=la)
#
#         meta_data_batch = []
#         for j in range(result.shape[0]):
#             if not pd.isnull(result.iloc[j, i]):  # set nan to 0?
#                 meta_data_batch.append({'X_train': data['X_train'],
#                                         'y_train': data['y_train'],
#                                         'dataset_idx': i,
#                                         'la': la,
#                                         'components': components_df_index.iloc[j, :].values,
#                                         'performance': result.iloc[j, i]})
#         meta_data.append(meta_data_batch)
#
# # initialization
# utils.set_seed(42)
# model = meta_predictor(n_col=components_df_index.shape[1],
#                        n_per_col=[max(components_df_index.iloc[:, i])+1
#                                   for i in range(components_df_index.shape[1])])
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
# # fitting
# fit(meta_data, model, optimizer)
#
# # testing
# data_generator = DataGenerator(dataset=self.test_dataset, seed=1)
# test_data = data_generator.generator(la=test_la)
#
# with torch.no_grad():
#     _, _, pred = model([torch.from_numpy(test_data['X_train']).float() for i in range(components_df_index.shape[0])],
#                        [torch.from_numpy(test_data['y_train']).float() for i in range(components_df_index.shape[0])],
#                         torch.tensor([test_la for i in range(components_df_index.shape[0])]).unsqueeze(1),
#                         torch.from_numpy(components_df_index.values).float())


# def dataloader(meta_data, start_idx=None, end_idx=None, downsample=True):
#     X_list, y_list, la_list, components, targets = [], [], [], [], []
#     # for _ in meta_data[start_idx: end_idx]:
#     for _ in meta_data:
#         X_train = _['X_train']
#         y_train = _['y_train']
#         if downsample:
#             if X_train.shape[0] > 100:
#                 idx = np.random.choice(np.arange(X_train.shape[0]), 100, replace=False)
#                 X_train = X_train[idx, :]
#                 y_train = y_train[idx]
#             if X_train.shape[1] > 100:
#                 idx = np.random.choice(np.arange(X_train.shape[1]), 100, replace=False)
#                 X_train = X_train[:, idx]
#
#         X_list.append(torch.from_numpy(X_train).float())
#         y_list.append(torch.from_numpy(y_train).float())
#         la_list.append(_['la'])
#         components.append(_['components'])
#         targets.append(_['performance'])
#
#     la_list = torch.tensor(la_list).unsqueeze(1)
#     components = torch.from_numpy(np.stack(components)).float()
#     targets = torch.tensor(targets).float()
#
#     return X_list, y_list, la_list, components, targets


