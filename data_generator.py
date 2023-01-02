import numpy as np
import os
from math import ceil
from sklearn.model_selection import train_test_split
from metaod.models.gen_meta_features import generate_meta_features

from utils import Utils

class DataGenerator():
    def __init__(self,
                 seed:int=42,
                 dataset:str=None,
                 test_size:float=0.3,
                 generate_duplicates=False,
                 n_samples_lower_bound=None,
                 n_samples_upper_bound=None,
                 verbose=False):
        '''
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_lower_bound: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_lower_bound will be dropped
        :param n_samples_upper_bound: threshold for downsampling input samples, considering the computational cost
        '''

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        self.generate_duplicates = generate_duplicates
        self.n_samples_lower_bound = n_samples_lower_bound
        self.n_samples_upper_bound = n_samples_upper_bound

        # dataset list
        self.dataset_list = [os.path.splitext(_)[0] for _ in os.listdir('datasets')
                             if os.path.splitext(_)[1] == '.npz'] # classical AD datasets

        # myutils function
        self.utils = Utils()

        self.verbose = verbose

    def generator(self,
                  X=None,
                  y=None,
                  la=None,
                  at_least_one_labeled=False,
                  meta=False):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        '''

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
        else:
            data = np.load(os.path.join('datasets', self.dataset + '.npz'), allow_pickle=True)
            X = data['X']
            y = data['y']

        # if the dataset is too small, generating duplicate smaples up to n_samples_lower_bound
        if len(y) < self.n_samples_lower_bound and self.generate_duplicates:
            if self.verbose:
                print(f'generating duplicate samples for dataset {self.dataset}...')
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_lower_bound, replace=True)
            X = X[idx_duplicate]
            y = y[idx_duplicate]

        # if the dataset is too large, subsampling for considering the computational cost
        if len(y) > self.n_samples_upper_bound:
            if self.verbose:
                print(f'subsampling for dataset {self.dataset}...')
            self.utils.set_seed(self.seed)
            idx_sample = np.random.choice(np.arange(len(y)), self.n_samples_upper_bound, replace=False)
            X = X[idx_sample]
            y = y[idx_sample]

        # spliting the current data to the training set and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if type(la) == float:
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
        elif type(la) == int:
            if la > len(idx_anomaly):
                raise AssertionError(f'the number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !')
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        else:
            raise NotImplementedError

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)

        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        # generate meta-feature if necessary
        if meta:
            meta_features, _ = generate_meta_features(X_train)
        else:
            meta_features = None

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'meta_features': meta_features}