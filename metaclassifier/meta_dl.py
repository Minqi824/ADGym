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
# accelerate for same dataset and seed

import time
import os
import sys; sys.path.append('..')
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
                 seed:int =42,
                 metric: str='AUCPR',
                 suffix: str='',
                 grid_mode: int='small',
                 grid_size: int=1000,
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

        self.device = self.utils.get_device() # get device for gpu acceleration
        self.lr = 1e-2 # learning rate for meta classifier

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
    # add some constraints to improve the training process of meta classifier, e.g.,
    # we can remove some unimportant components after detailed analysis
    ############################## meta classifier of two-stage version ##############################
    def meta_fit(self):
        # set seed for reproductive results
        self.utils.set_seed(self.seed)
        # generate training data for meta predictor
        meta_features, las, components, performances = [], [], [], []

        for la in [5, 10, 25, 50]:
            result = pd.read_csv('../result/components/result-' + self.metric + '-test-' + '-'.join(
                [self.suffix, str(la), self.grid_mode, str(self.grid_size), 'GAN', str(self.gan_specific), str(self.seed)]) + '.csv')
            result.rename(columns={'Unnamed: 0': 'Components'}, inplace=True)

            # remove dataset of testing task
            result.drop([self.test_dataset], axis=1, inplace=True)
            assert self.test_dataset not in result.columns

            # using the rank ratio as target
            for i in range(1, result.shape[1]):
                r = np.argsort(np.argsort(-result.iloc[:, i].fillna(0).values))
                result.iloc[:, i] = r / result.shape[0]

            # transform result dataframe for preparation
            self.components_df_index = self.components_process(result)

            for i in range(result.shape[0]):
                for j in range(1, result.shape[1]):
                    if not pd.isnull(result.iloc[i, j]) and result.columns[j] != self.test_dataset:  # set nan to 0?
                        meta_feature = np.load(
                            '../datasets/meta-features/' + 'meta-features-' + result.columns[j] + '-' + str(
                                la) + '-' + str(self.seed) + '.npz', allow_pickle=True)

                        # preparing training data for meta classifier
                        # note that we only extract meta features in training set of both training & testing tasks
                        meta_features.append(meta_feature['data'])
                        las.append(la)
                        components.append(self.components_df_index.iloc[i, :].values)
                        performances.append(result.iloc[i, j])

        del la

        meta_features = np.stack(meta_features)
        components = np.stack(components)

        # fillna in extracted meta-features
        meta_features = pd.DataFrame(meta_features).fillna(0).values
        # min-max scaling for meta-features
        self.scaler_meta_features = MinMaxScaler(clip=True).fit(meta_features)
        meta_features = self.scaler_meta_features.transform(meta_features)
        # min-max scaling for la
        las = np.array(las).reshape(-1, 1)
        self.scaler_las = MinMaxScaler(clip=True).fit(las)
        las = self.scaler_las.transform(las)

        # to tensor
        meta_features = torch.from_numpy(meta_features).float().to(self.device)
        las = torch.from_numpy(las.squeeze()).float().to(self.device)
        components = torch.from_numpy(components).float().to(self.device)
        performances = torch.tensor(performances).float().to(self.device)

        # to dataloader
        train_loader = DataLoader(TensorDataset(meta_features, las, components, performances),
                                  batch_size=512, shuffle=True, drop_last=True)

        # initialize meta classifier
        self.model = meta_predictor(n_col=components.size(1),
                                    n_per_col=[max(components[:, i]).item() + 1 for i in range(components.size(1))])
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # fitting meta classifier
        print('fitting meta classifier...')
        fit(train_loader, self.model, optimizer)

        return self

    @torch.no_grad()
    def meta_predict(self):
        # 1. meta-feature for testing dataset
        meta_feature_test = np.load(
            '../datasets/meta-features/' + 'meta-features-' + self.test_dataset + '-' + str(self.test_la) + '-' + str(self.seed) + '.npz',
            allow_pickle=True)
        meta_feature_test = meta_feature_test['data']
        meta_feature_test = np.stack([meta_feature_test for i in range(self.components_df_index.shape[0])])
        meta_feature_test = pd.DataFrame(meta_feature_test).fillna(0).values
        meta_feature_test = self.scaler_meta_features.transform(meta_feature_test)
        meta_feature_test = torch.from_numpy(meta_feature_test).float().to(self.device)

        # 2. number of labeled anomalies in testing dataset
        la_test = np.repeat(self.test_la, self.components_df_index.shape[0]).reshape(-1, 1)
        la_test = self.scaler_las.transform(la_test)
        la_test = torch.from_numpy(la_test.squeeze()).float().to(self.device)

        # 3. components (predefined)
        components_test = torch.from_numpy(self.components_df_index.values).float().to(self.device)

        # predicting
        _, pred = self.model(meta_feature_test, la_test.unsqueeze(1), components_test)
        pred = pred.cpu()

        # since we have already train-test all the components on each dataset,
        # we can only inquire the experiment result with no information leakage
        result = pd.read_csv('../result/components/result-' + self.metric + '-test-' + '-'.join(
            [self.suffix, str(self.test_la), self.grid_mode, str(self.grid_size), 'GAN',
             str(self.gan_specific), str(self.seed)]) + '.csv')
        # for _ in torch.argsort(-pred.squeeze()):
        for _ in torch.argsort(pred.squeeze()):
            pred_performance = result.loc[_.item(), self.test_dataset]
            if not pd.isnull(pred_performance):
                break

        return pred_performance

    ############################## meta classifier of end-to-end version ##############################
    # dataloader for end2end meta classifier version
    def dataloader(self, meta_data, downsample=True, n_samples_upper_bound=256, n_features_upper_bound=100):
        self.utils.set_seed(self.seed)

        X_list, y_list, la_list, components, targets = [], [], [], [], []
        for _ in meta_data:
            X_train = _['X_train']
            y_train = _['y_train']
            if downsample:
                if X_train.shape[0] > n_samples_upper_bound:
                    idx = np.random.choice(np.arange(X_train.shape[0]), n_samples_upper_bound, replace=False)
                    X_train = X_train[idx, :]
                    y_train = y_train[idx]

                if X_train.shape[1] > n_features_upper_bound:
                    idx = np.random.choice(np.arange(X_train.shape[1]), n_features_upper_bound, replace=False)
                    X_train = X_train[:, idx]

            X_list.append(torch.from_numpy(X_train).float().to(self.device))
            y_list.append(torch.from_numpy(y_train).float().to(self.device))
            la_list.append(_['la'])
            components.append(_['components'])
            targets.append(_['performance'])

        la_list = torch.tensor(la_list).unsqueeze(1).to(self.device)
        components = torch.from_numpy(np.stack(components)).float().to(self.device)
        targets = torch.tensor(targets).float().to(self.device)

        return [X_list, y_list, la_list, components, targets]

    def meta_fit_end2end(self):
        # set seed for reproductive results
        self.utils.set_seed(self.seed)

        meta_data = []
        for la in [5, 10, 25, 50]:
            result = pd.read_csv('../result/components/result-' + self.metric + '-test-' + '-'.join(
                [self.suffix, str(la), self.grid_mode, str(self.grid_size), 'GAN', str(self.gan_specific), str(self.seed)]) + '.csv')
            result.rename(columns={'Unnamed: 0': 'Components'}, inplace=True)

            # remove dataset of testing task
            result.drop([self.test_dataset], axis=1, inplace=True)
            assert self.test_dataset not in result.columns

            # using the rank ratio as target
            for i in range(1, result.shape[1]):
                r = np.argsort(np.argsort(-result.iloc[:, i].fillna(0).values))
                result.iloc[:, i] = r / result.shape[0]

            # transform result dataframe for preparation
            self.components_df_index = self.components_process(result)

            # meta data batch
            for i in range(1, result.shape[1]):
                # generate dataset
                self.data_generator.dataset = result.columns[i]
                self.data_generator.seed = self.seed
                data = self.data_generator.generator(la=la)

                meta_data_batch = []
                for j in range(result.shape[0]):
                    if not pd.isnull(result.iloc[j, i]):  # set nan to 0?
                        meta_data_batch.append({'X_train': data['X_train'],
                                                'y_train': data['y_train'],
                                                'dataset_idx': i,
                                                'la': la,
                                                'components': self.components_df_index.iloc[j, :].values,
                                                'performance': result.iloc[j, i]})
                if len(meta_data_batch) > 0:
                    meta_data.append(self.dataloader(meta_data_batch))

            # initialize meta classifier
            self.model = meta_predictor_end2end(n_col=self.components_df_index.shape[1],
                                                n_per_col=[max(self.components_df_index.iloc[:, i]) + 1 for i in
                                                           range(self.components_df_index.shape[1])])
            self.model.to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            # fitting meta classifier
            fit_end2end(meta_data, self.model, optimizer)

            return self

    @torch.no_grad()
    def meta_predict_end2end(self):
        self.data_generator.dataset = self.test_dataset
        self.data_generator.seed = self.seed
        test_data = self.data_generator.generator(la=self.test_la)

        # notice that we can only use the training set of the testing task
        # X_list_test = [torch.from_numpy(test_data['X_train']).float() for i in range(self.components_df_index.shape[0])]
        # y_list_test = [torch.from_numpy(test_data['y_train']).float() for i in range(self.components_df_index.shape[0])]
        # la_test = torch.tensor([self.test_la for i in range(self.components_df_index.shape[0])]).unsqueeze(1)
        # components_test = torch.from_numpy(self.components_df_index.values).float()
        #
        # _, _, pred = self.model(X_list_test, y_list_test, la_test, components_test)

        preds = []
        for i in range(self.components_df_index.shape[0]):
            X_list_test = [torch.from_numpy(test_data['X_train']).float().to(self.device)]
            y_list_test = [torch.from_numpy(test_data['y_train']).float().to(self.device)]
            la_test = torch.tensor([[self.test_la]]).to(self.device)
            components_test = torch.from_numpy(self.components_df_index.values[i, :].reshape(1, -1)).float().to(self.device)
            _, _, pred = self.model(X_list_test, y_list_test, la_test, components_test)
            preds.append(pred.cpu().item())
        preds = np.array(preds)

        # since we have already train-test all the components on each dataset,
        # we can only inquire the experiment result with no information leakage
        result = pd.read_csv('../result/components/result-' + self.metric + '-test-' + '-'.join(
            [self.suffix, str(self.test_la), self.grid_mode, str(self.grid_size), 'GAN',
             str(self.gan_specific), str(self.seed)]) + '.csv')
        # for _ in np.argsort(-preds):
        for _ in np.argsort(preds):
            pred_performance = result.loc[_, self.test_dataset]
            if not pd.isnull(pred_performance):
                break

        return pred_performance

# demo for debugging
def run_demo():
    run_meta = meta(seed=1,
                    metric='AUCPR',
                    suffix='formal',
                    grid_mode='small',
                    grid_size=1000,
                    gan_specific=False,
                    test_dataset='44_Wilt')

    clf = run_meta.meta_fit()
    clf.test_la = 25
    perf = clf.meta_predict()
    print(perf)

    # clf = run_meta.meta_fit_end2end()
    # clf.test_la = 10
    # perf = clf.meta_predict_end2end()
    # print(perf)

# experiments for two-stage or end-to-end version of meta classifer
def run(suffix, grid_mode, grid_size, gan_specific, mode):
    # run experiments for comparing proposed meta classifier and current SOTA methods
    utils = Utils()

    for metric in ['AUCROC', 'AUCPR']:
        # result of current SOTA models
        result_SOTA_semi = pd.read_csv('../result/' + metric + '-SOTA-semi-supervise.csv')
        result_SOTA_sup = pd.read_csv('../result/' + metric + '-SOTA-supervise.csv')
        result_SOTA = result_SOTA_semi.merge(result_SOTA_sup, how='inner', on='Unnamed: 0')
        del result_SOTA_semi, result_SOTA_sup

        meta_baseline_rs_performance = np.repeat(-1, result_SOTA.shape[0]).astype(float)
        meta_baseline_ss_performance = np.repeat(-1, result_SOTA.shape[0]).astype(float)
        meta_baseline_gt_performance = np.repeat(-1, result_SOTA.shape[0]).astype(float)
        meta_classifier_performance = np.repeat(-1, result_SOTA.shape[0]).astype(float)

        for i in tqdm(range(result_SOTA.shape[0])):
            # extract the testing task from the SOTA model results
            test_dataset, test_seed, test_la = ast.literal_eval(result_SOTA.iloc[i, 0])
            print(f'Experiments on meta classifier: Dataset: {test_dataset}, seed: {test_seed}, la: {test_la}')

            # set seed for reproductive results
            utils.set_seed(test_seed)

            # result of other meta baseline, including:
            # 1. rs: random selection;
            # 2. ss: selection based on the labeled anomalies in the training set of testing task
            # 3. gt: ground truth where the best model can always be selected
            result_meta_baseline_train = pd.read_csv('../result/components/result-' + metric + '-train-' + '-'.join(
                [suffix, str(test_la), grid_mode, str(grid_size), 'GAN', str(gan_specific), str(test_seed)]) + '.csv')
            result_meta_baseline_test = pd.read_csv('../result/components/result-' + metric + '-test-' + '-'.join(
                [suffix, str(test_la), grid_mode, str(grid_size), 'GAN', str(gan_specific), str(test_seed)]) + '.csv')

            # random search
            for _ in range(result_meta_baseline_train.shape[0]):
                idx = np.random.choice(np.arange(result_meta_baseline_train.shape[0]), 1).item()
                perf = result_meta_baseline_test.loc[idx, test_dataset]
                if not pd.isnull(perf):
                    meta_baseline_rs_performance[i] = perf; del perf
                    break

            # select the best components based on the performance in the training set of testing task (i.e., test dataset)
            for _ in np.argsort(-result_meta_baseline_train.loc[:, test_dataset].values):
                perf = result_meta_baseline_test.loc[_, test_dataset]
                if not pd.isnull(perf):
                    meta_baseline_ss_performance[i] = perf; del perf
                    break

            # ground truth
            perf = np.max(result_meta_baseline_test.loc[:, test_dataset])
            meta_baseline_gt_performance[i] = perf; del perf

            result_SOTA['Meta_baseline_rs'] = meta_baseline_rs_performance
            result_SOTA['Meta_baseline_ss'] = meta_baseline_ss_performance
            result_SOTA['Meta_baseline_gt'] = meta_baseline_gt_performance

            # run meta classifier
            run_meta = meta(seed=test_seed,
                            metric=metric,
                            suffix=suffix,
                            grid_mode=grid_mode,
                            grid_size=grid_size,
                            gan_specific=gan_specific,
                            test_dataset=test_dataset)

            try:
                if mode == 'two-stage':
                    # retrain the meta classifier if we need to test on the new testing task
                    if i == 0 or test_dataset != test_dataset_previous or test_seed != test_seed_previous:
                        clf = run_meta.meta_fit()
                    else:
                        print('Using the trained meta classifier to predict...')

                    clf.test_la = test_la
                    perf = clf.meta_predict()

                elif mode == 'end-to-end':
                    # retrain the meta classifier if we need to test on the new testing task
                    if i == 0 or test_dataset != test_dataset_previous or test_seed != test_seed_previous:
                        clf = run_meta.meta_fit_end2end()
                    else:
                        print('Using the trained meta classifier to predict...')

                    clf.test_la = test_la
                    perf = clf.meta_predict_end2end()

                else:
                    raise NotImplementedError

                meta_classifier_performance[i] = perf
            except Exception as error:
                print(f'Something error when training meta-classifier: {error}')
                meta_classifier_performance[i] = -1

            result_SOTA['Meta'] = meta_classifier_performance

            if mode == 'two-stage':
                result_SOTA.to_csv('../result/' + metric + '-meta-dl-twostage.csv', index=False)
            elif mode == 'end-to-end':
                result_SOTA.to_csv('../result/' + metric + '-meta-dl-end2end.csv', index=False)
            else:
                raise NotImplementedError

            test_dataset_previous = test_dataset
            test_seed_previous = test_seed

# formal experiments
run(suffix='formal', grid_mode='small', grid_size=1000, gan_specific=False, mode='two-stage')
# run(suffix='formal', grid_mode='small', grid_size=1000, gan_specific=False, mode='end-to-end')


# demo experiment for debugging
# run_demo()