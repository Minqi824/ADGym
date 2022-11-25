import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from iteration_utilities import unique_everseen
import time

from data_generator import DataGenerator
from components import Components

# dataset + lr +seed

class ADGym():
    def __init__(self, la=0.10, suffix='', grid_mode='small', grid_size=100):
        self.la = la
        self.suffix = suffix + '_' + str(la) + '_' + grid_mode + '_' + str(grid_size)
        self.seed_list = list(np.arange(3) + 1)

        self.grid_mode = grid_mode
        self.grid_size = grid_size

        if isinstance(la, int):
            self.mode = 'nla'
        elif isinstance(la, float):
            self.mode = 'rla'
        else:
            raise NotImplementedError

        self.generate_duplicates = False
        self.n_samples_threshold = 1000
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_threshold=self.n_samples_threshold)

    def dataset_filter(self, dataset_list_org):
        dataset_list = []
        dataset_size = []
        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset
                data = self.data_generator.generator(la=1.00, at_least_one_labeled=True)

                if not self.generate_duplicates and len(data['y_train']) + len(
                        data['y_test']) < self.n_samples_threshold:
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

    def generate_gyms(self):
        # generator combinations of different components
        com = Components()
        print(com.gym(mode=self.grid_mode))

        gyms_comb = list(product(*list(com.gym().values())))
        keys = list(com.gym().keys())
        gyms = []

        for _ in tqdm(gyms_comb):
            gym = {}
            for j, __ in enumerate(_):
                gym[keys[j]] = __

            if gym['layers'] != len(gym['hidden_size_list']):
                continue

            if gym['loss_name'] == 'inverse' and gym['batch_resample']:
                continue

            if gym['loss_name'] != 'inverse' and not gym['batch_resample']:
                continue

            # delete ResNet & FTT ReLU: activation layers
            if gym['network_architecture'] in ['ResNet', 'FTT']:
                if gym['act_fun'] != 'ReLU':
                    continue

            # delete FTT: hidden_size_list, drop out
            if gym['network_architecture'] == 'FTT':
                gym['hidden_size_list'] = None
                gym['dropout'] = None

            gyms.append(gym)

        # random selection
        if len(gyms) > self.grid_size:
            idx = np.random.choice(np.arange(len(gyms)), self.grid_size, replace=False)
            gyms = [gyms[_] for _ in idx]
        # remove duplicates
        gyms = list(unique_everseen(gyms))

        return gyms

    def run(self):
        # datasets
        dataset_list = [os.path.splitext(_)[0] for _ in os.listdir('datasets') if os.path.splitext(_)[1] == '.npz']
        dataset_list = self.dataset_filter(dataset_list)

        # gyms
        gyms = self.generate_gyms()

        # save results
        df_results_AUCROC = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
        df_results_AUCPR = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
        df_results_runtime = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)

        for dataset in dataset_list:
            # generate data
            data_generator = DataGenerator(dataset=dataset)
            data = data_generator.generator(la=self.la)

            for gym in tqdm(gyms):
                aucroc_list, aucpr_list, time_list = [], [], []
                for seed in self.seed_list:
                    com = Components(seed=seed,
                                     data=data,
                                     augmentation=gym['augmentation'],
                                     preprocess=gym['preprocess'],
                                     network_architecture=gym['network_architecture'],
                                     layers=gym['layers'],
                                     hidden_size_list=gym['hidden_size_list'],
                                     act_fun=gym['act_fun'],
                                     dropout=gym['dropout'],
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

                        # predicting
                        metrics = com.f_predict_score()
                        aucroc_list.append(metrics['aucroc'])
                        aucpr_list.append(metrics['aucpr'])
                        time_list.append(end_time - start_time)
                    except:
                        print(f'Dataset: {dataset}, Current combination: {gym}, training failure.')
                        aucroc_list.append(None)
                        aucpr_list.append(None)
                        time_list.append(None)
                        pass
                        continue

                # save results
                if all([_ is not None for _ in aucroc_list]) and all([_ is not None for _ in aucpr_list]) and all([_ is not None for _ in time_list]):
                    df_results_AUCROC.loc[str(gym), dataset] = np.mean(aucroc_list)
                    df_results_AUCPR.loc[str(gym), dataset] = np.mean(aucpr_list)
                    df_results_runtime.loc[str(gym), dataset] = np.mean(time_list)
                print(f'Dataset: {dataset}, Current combination: {gym}, training sucessfully.')

                # output
                if not os.path.exists('result'):
                    os.makedirs('result')
                df_results_AUCROC.to_csv(os.path.join('result', 'result_AUCROC' + self.suffix + '.csv'), index=True)
                df_results_AUCPR.to_csv(os.path.join('result', 'result_AUCPR' + self.suffix + '.csv'), index=True)
                df_results_runtime.to_csv(os.path.join('result', 'result_runtime' + self.suffix + '.csv'), index=True)


adgym = ADGym(la=0.1, grid_mode='small', grid_size=1000)
adgym.run()