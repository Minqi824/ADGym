import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from iteration_utilities import unique_everseen
import time

from data_generator import DataGenerator
from components import Components

# generator combinations of different components
com = Components()
print(com.gym())

gyms_comb = list(product(*list(com.gym().values())))
keys = list(com.gym().keys())
gyms = []

for _ in tqdm(gyms_comb):
    gym = {}
    for j, __ in enumerate(_):
        gym[keys[j]] = __

    if gym['layers'] != len(gym['hidden_size_list']):
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
idx = np.random.choice(np.arange(len(gyms)), 100, replace=False)
gyms = [gyms[_] for _ in idx]
# remove duplicates
gyms = list(unique_everseen(gyms))


# datasets
dataset_list = [os.path.splitext(_)[0] for _ in os.listdir('datasets') if os.path.splitext(_)[1] == '.npz']

df_results_AUCROC = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
df_results_AUCPR = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)
df_results_runtime = pd.DataFrame(data=None, index=[str(_) for _ in gyms], columns=dataset_list)

for dataset in dataset_list:
    # generate data
    data_generator = DataGenerator(dataset=dataset)
    data = data_generator.generator(la=0.20)

    for gym in tqdm(gyms):
        com = Components(data=data,
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

            # save results
            df_results_AUCROC.loc[str(gym), dataset] = metrics['aucroc']
            df_results_AUCPR.loc[str(gym), dataset] = metrics['aucpr']
            df_results_runtime.loc[str(gym), dataset] = end_time - start_time
            print(f'Dataset: {dataset}, Current combination: {gym}, training sucessfully.')

            # output
            df_results_AUCROC.to_csv('result_AUCROC.csv', index=True)
            df_results_AUCPR.to_csv('result_AUCPR.csv', index=True)
            df_results_runtime.to_csv('result_runtime.csv', index=True)

        except:
            print(f'Dataset: {dataset}, Current combination: {gym}, training failure.')
            pass
            continue