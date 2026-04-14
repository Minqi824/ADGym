import os
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
from keras import backend as K

from data_generator import DataGenerator
from utils import Utils

# most of the codes are from the ADBench (NeurIPS 2022)
# notice that the compared SOTA models should use the same training settings in ADGym
class RunPipeline():
    def __init__(self,
                 suffix: str=None,
                 mode: str='rla',
                 parallel: str=None,
                 generate_duplicates: bool=False,
                 n_samples_lower_bound: int=1000,
                 n_samples_upper_bound: int=3000):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_lower_bound: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_lower_bound will be dropped
        :param n_samples_upper_bound: threshold for downsampling input samples, considering the computational cost
        '''

        # utils function
        self.utils = Utils()
        self.mode = mode
        self.parallel = parallel

        # global parameters
        self.generate_duplicates = generate_duplicates
        self.n_samples_lower_bound = n_samples_lower_bound
        self.n_samples_upper_bound = n_samples_upper_bound


        # the suffix of all saved files
        self.suffix = suffix + '-' + self.parallel

        if not os.path.exists('result'):
            os.makedirs('result')

        # data generator instantiation
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_lower_bound=self.n_samples_lower_bound,
                                            n_samples_upper_bound=self.n_samples_upper_bound)

        if self.parallel != 'unsupervise':
            # ratio of labeled anomalies
            self.rla_list = [0.05, 0.10, 0.20]
            # number of labeled anomalies
            self.nla_list = [5, 10, 20]
        else:
            self.rla_list = [0.0]
            self.nla_list = [0]

        # seed list
        self.seed_list = list(np.arange(3) + 1)

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervise':
            from baseline.PyOD import PYOD
            from baseline.DAGMM.run import DAGMM

            # from pyod
            # for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
            #           'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL', 'DeepSVDD']:
            #     self.model_dict[_] = PYOD

            for _ in ['IForest', 'ECOD', 'DeepSVDD']:
                self.model_dict[_] = PYOD

            # DAGMM
            self.model_dict['DAGMM'] = DAGMM

        # semi-supervised algorithms
        elif self.parallel == 'semi-supervise':
            from baseline.PyOD import PYOD
            from baseline.GANomaly.run import GANomaly
            from baseline.DeepSAD.src.run import DeepSAD
            from baseline.REPEN.run import REPEN
            from baseline.DevNet.run import DevNet
            from baseline.PReNet.run import PReNet
            from baseline.FEAWAD.run import FEAWAD

            self.model_dict = {'GANomaly': GANomaly,
                               'REPEN': REPEN,
                               'DeepSAD': DeepSAD,
                               'DevNet': DevNet,
                               'PReNet': PReNet,
                               'FEAWAD': FEAWAD,
                               'XGBOD': PYOD}

        # fully-supervised algorithms
        elif self.parallel == 'supervise':
            from baseline.Supervised import supervised
            from baseline.FTTransformer.run import FTTransformer

            # from sklearn
            for _ in ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
                self.model_dict[_] = supervised
            # ResNet and FTTransformer for tabular data
            for _ in ['ResNet', 'FTTransformer']:
                self.model_dict[_] = FTTransformer

        else:
            raise NotImplementedError

        # We remove the following model for considering the computational cost
        for _ in ['SOGAAL', 'MOGAAL', 'LSCP', 'MCD', 'FeatureBagging']:
            if _ in self.model_dict.keys():
                self.model_dict.pop(_)

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = [os.path.splitext(_)[0] for _ in os.listdir('datasets')
                            if os.path.splitext(_)[1] == '.npz']

        dataset_list, dataset_size = [], []
        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset

                try:
                    data = self.data_generator.generator(la=1.00, at_least_one_labeled=True)
                except:
                    add = False
                    pass
                    continue

                if not self.generate_duplicates and len(data['y_train']) + len(data['y_test']) < self.n_samples_lower_bound:
                    add = False

                else:
                    if self.mode == 'nla' and sum(data['y_train']) >= self.nla_list[-1]:
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

    # model fitting function
    def model_fit(self):
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # fitting
            start_time = time.time()
            self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'])
            end_time = time.time(); time_fit = end_time - start_time


            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
            else:
                score_test = self.clf.predict_score(self.data['X_test'])
            end_time = time.time(); time_inference = end_time - start_time

            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_fit, time_inference, result

    # run the experiment
    def run(self):
        #  filteting dataset that does not meet the experimental requirements
        dataset_list = self.dataset_filter()

        # experimental parameters
        if self.mode == 'nla':
            experiment_params = list(product(dataset_list, self.seed_list, self.nla_list))
        else:
            experiment_params = list(product(dataset_list, self.seed_list, self.rla_list))

        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')

        # save the results
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))

        for i, params in tqdm(enumerate(experiment_params)):
            dataset, self.seed, la = params

            if self.parallel == 'unsupervise' and la != 0.0:
                continue

            print(f'Current experiment parameters: {params}')

            # generate data
            self.data_generator.seed = self.seed
            self.data_generator.dataset = dataset

            try:
                self.data = self.data_generator.generator(la=la, at_least_one_labeled=True)

            except Exception as error:
                print(f'Error when generating data: {error}')
                pass
                continue

            for model_name in tqdm(self.model_dict.keys()):
                self.model_name = model_name
                self.clf = self.model_dict[self.model_name]

                # fit model
                time_fit, time_inference, result = self.model_fit()

                # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                df_AUCROC[model_name].iloc[i] = result['aucroc']
                df_AUCPR[model_name].iloc[i] = result['aucpr']
                df_time_fit[model_name].iloc[i] = time_fit
                df_time_inference[model_name].iloc[i] = time_inference

                df_AUCROC.to_csv(os.path.join(os.getcwd(), 'result', 'AUCROC-' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.getcwd(), 'result', 'AUCPR-' + self.suffix + '.csv'), index=True)
                df_time_fit.to_csv(os.path.join(os.getcwd(), 'result', 'Time-fit-' + self.suffix + '.csv'), index=True)
                df_time_inference.to_csv(os.path.join(os.getcwd(), 'result', 'Time-inference-' + self.suffix + '.csv'), index=True)

# run the above pipeline for reproducing the results in the paper

pipeline = RunPipeline(suffix='SOTA', parallel='unsupervise', mode='nla')
pipeline.run()
