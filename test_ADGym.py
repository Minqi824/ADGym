import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from iteration_utilities import unique_everseen
import time
import gc
from keras import backend as K
from data_generator import DataGenerator
from utils import Utils
from components import Components

# TODO
# using the absolute anomaly score in minus, inverse and hinge loss?
# since these loss functions are originally designed for the representation learning,
# e.g., Euclidean distance or reconstruction errors, which is always positive

# batch normalization
# add other network architectures that may outperform the tree-based methods, e.g., MoE or TabNet


class ADGym():
    def __init__(self, seed_list: list=[1, 2, 3], la=5, suffix='',
                 grid_mode='small', grid_size=100, gan_specific=False, dataset_list=None):
        '''
        :param la: number of labeled anomalies
        :param suffix: suffix for save experimental results
        :param grid_mode: use large or small scale of combinations
        :param grid_size: whether to sampling grids to save computational cost
        :param gan_specific: whether to specific GAN-based data augmentation method (which is time-consuming)
        '''
        self.la = la
        self.suffix = '-'.join([suffix, str(la), grid_mode, str(grid_size), 'GAN', str(gan_specific)])
        self.seed_list = seed_list

        self.grid_mode = grid_mode
        self.grid_size = grid_size
        self.gan_specific = gan_specific

        self.dataset_list = dataset_list
        self.utils = Utils()

        if isinstance(la, int):
            self.mode = 'nla'
        elif isinstance(la, float):
            self.mode = 'rla'
        else:
            raise NotImplementedError

        self.generate_duplicates = False # whether to generate duplicates for small datasets
        self.n_samples_lower_bound = 1000 # lower bound of sample size
        self.n_samples_upper_bound = 3000 # upper bound of sample size
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_lower_bound=self.n_samples_lower_bound,
                                            n_samples_upper_bound=self.n_samples_upper_bound)

    # filtering out datasets that do not meet the requirements
    def dataset_filter(self, dataset_list_org):
        dataset_list, dataset_size = [], []
        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.dataset = dataset
                self.data_generator.seed = seed

                try:
                    data = self.data_generator.generator(la=1.00, at_least_one_labeled=True)
                except:
                    add = False
                    pass
                    continue

                if not self.generate_duplicates and \
                        len(data['y_train']) + len(data['y_test']) < self.n_samples_lower_bound:
                    add = False

                else:
                    if self.mode == 'nla' and sum(data['y_train']) >= self.la:
                        pass

                    elif self.mode == 'rla' and sum(data['y_train']) > 0:
                        pass

                    else:
                        add = False

            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            else:
                print(f"remove the dataset {dataset}")

        # sort datasets by their sample size
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]

        return dataset_list

    def generate_gyms(self):
        self.utils.set_seed(42)
        # generate combinations of different components
        com = Components(gan_specific=self.gan_specific)
        print(com.gym(mode=self.grid_mode)) # see the entire components in the current grid mode (either large or small)

        gyms_comb = list(product(*list(com.gym(mode=self.grid_mode).values())))
        keys = list(com.gym(mode=self.grid_mode).keys())
        gyms = []

        for _ in tqdm(gyms_comb):
            gym = {} # save components in dict
            for j, __ in enumerate(_):
                gym[keys[j]] = __

            if gym['layers'] != len(gym['hidden_size_list']):
                continue

            # for inverse loss, we do not perform batch resampling strategy
            if gym['loss_name'] == 'inverse' and gym['batch_resample']:
                continue

            # for other loss functions, we use batch resampling strategy
            if gym['loss_name'] != 'inverse' and not gym['batch_resample']:
                continue

            # ToDo: ordinal loss for other network architectures
            if gym['loss_name'] == 'ordinal' and gym['network_architecture'] != 'MLP':
                continue

            # delete components of network architecture = ResNet or FTT while the activation function is not RELU
            if gym['network_architecture'] in ['ResNet', 'FTT']:
                if gym['act_fun'] != 'ReLU':
                    continue

            # delete FTT: hidden_size_list, drop out
            if gym['network_architecture'] == 'FTT':
                gym['hidden_size_list'] = None
                gym['dropout'] = None

            gyms.append(gym)

        # random selection for considering computational cost
        if len(gyms) > self.grid_size:
            idx = np.random.choice(np.arange(len(gyms)), self.grid_size, replace=False)
            gyms = [gyms[_] for _ in idx]
        # remove duplicated components
        gyms = list(unique_everseen(gyms))

        return gyms

    def run(self):
        if self.dataset_list is None:
            # dataset list
            dataset_list = [os.path.splitext(_)[0] for _ in os.listdir('datasets') if os.path.splitext(_)[1] == '.npz']
        else:
            dataset_list = self.dataset_list

        # filtering dataset
        dataset_list = self.dataset_filter(dataset_list)
        # generate components
        gyms = self.generate_gyms()

        # create save path
        if not os.path.exists('datasets/meta-features'):
            os.makedirs('datasets/meta-features')
        if not os.path.exists('result'):
            os.makedirs('result')

        for seed in self.seed_list:
            # save results
            df_results_AUCROC_train = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
            df_results_AUCROC_test = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
            df_results_AUCPR_train = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
            df_results_AUCPR_test = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
            df_results_runtime = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
            for dataset in dataset_list:
                # data generator instantiation
                self.data_generator.dataset = dataset
                self.data_generator.seed = seed
                for j, gym in tqdm(enumerate(gyms)):
                    # generate data and save meta-features
                    if j == 0:
                        data = self.data_generator.generator(la=self.la, meta=True)
                        np.savez_compressed(os.path.join('datasets/meta-features', 'meta-features-' + dataset +
                                                         '-' + str(self.la) + '-' + str(seed) + '.npz'),
                                            data=data['meta_features'])

                    com = Components(seed=seed,
                                     data=data,
                                     augmentation=gym['augmentation'],
                                     preprocess=gym['preprocess'],
                                     network_architecture=gym['network_architecture'],
                                     layers=gym['layers'],
                                     hidden_size_list=gym['hidden_size_list'],
                                     act_fun=gym['act_fun'],
                                     dropout=gym['dropout'],
                                     network_initialization=gym['network_initialization'],
                                     training_strategy=gym['training_strategy'],
                                     loss_name=gym['loss_name'],
                                     optimizer_name=gym['optimizer_name'],
                                     batch_resample=gym['batch_resample'],
                                     epochs=gym['epochs'],
                                     batch_size=gym['batch_size'],
                                     lr=gym['lr'],
                                     weight_decay=gym['weight_decay'])

                    try:
                        # training
                        start_time = time.time()
                        com.f_train()
                        end_time = time.time()
                        fit_time = end_time - start_time

                        # predicting
                        metrics_train, metrics_test = com.f_predict_score()

                        print(f'Dataset: {dataset}, Current combination: {gym}, training successfully.'
                              f' Performance (train): {metrics_train}, Performance (test): {metrics_test}')

                    except Exception as error:
                        print(f'Dataset: {dataset}, Current combination: {gym}, training failure. Error: {error}')
                        metrics_train, metrics_test, fit_time = None, None, None
                        pass
                        continue

                    K.clear_session()
                    del com
                    gc.collect()

                    # save results
                    if metrics_train is not None and metrics_test is not None and fit_time is not None:
                        df_results_AUCROC_train.loc[str(gym), dataset] = metrics_train['aucroc']
                        df_results_AUCROC_test.loc[str(gym), dataset] = metrics_test['aucroc']
                        df_results_AUCPR_train.loc[str(gym), dataset] = metrics_train['aucpr']
                        df_results_AUCPR_test.loc[str(gym), dataset] = metrics_test['aucpr']
                        df_results_runtime.loc[str(gym), dataset] = fit_time
                        print(f'Dataset: {dataset}, Current combination: {gym}, training successfully.')
                    else:
                        print(f'Dataset: {dataset}, Current combination: {gym}, training failure.')

                    # output
                    df_results_AUCROC_train.to_csv(os.path.join('result', 'result-AUCROC-train-' + self.suffix + '-' + str(seed) + '.csv'), index=True)
                    df_results_AUCROC_test.to_csv(os.path.join('result', 'result-AUCROC-test-' + self.suffix + '-' + str(seed) + '.csv'), index=True)
                    df_results_AUCPR_train.to_csv(os.path.join('result', 'result-AUCPR-train-' + self.suffix + '-' + str(seed) + '.csv'), index=True)
                    df_results_AUCPR_test.to_csv(os.path.join('result', 'result-AUCPR-test-' + self.suffix + '-' + str(seed) + '.csv'), index=True)
                    df_results_runtime.to_csv(os.path.join('result', 'result-runtime-' + self.suffix + '-' + str(seed) + '.csv'), index=True)

                del data

adgym = ADGym(suffix='formal', la=25, grid_mode='small', grid_size=3, gan_specific=False)
adgym.run()