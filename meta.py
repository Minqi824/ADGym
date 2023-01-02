# training a meta classifier to predict detection performance by given:
# 1. meta-feature (either use end-to-end training strategy, e.g., dataset2vec or use metaod to extract)
# 2. number of labeled anomalies
# 3. network components

# To Do
# end-to-end meta-feature (should we follow a pretrained-finetune process?)

import time
import os
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
data_generator = DataGenerator()

from utils import Utils
utils = Utils()


def components_process(result):
    # 只显示不一样的components
    Components_list = [ast.literal_eval(_) for _ in result['Components']]
    keys = list(ast.literal_eval(result['Components'][0]).keys())
    keys_show = []

    for k in keys:
        options = [str(_[k]) for _ in Components_list if _[k] is not None]
        if len(set(options)) == 1 or len(options) == 0:
            continue
        else:
            keys_show.append(k)

    # 能否学习components? (学习一个简单的classifier)
    components_list = []
    for c in Components_list:
        components_list.append({k: c[k] for k in keys_show})
    components_df = pd.DataFrame(components_list)
    components_df = components_df.replace([None], 'None')
    components_df_index = components_df.copy()

    for col in components_df_index.columns:
        components_df_index[col] = preprocessing.LabelEncoder().fit_transform(components_df_index[col])

    return components_df_index

def dataloader(meta_data, start_idx=None, end_idx=None, downsample=True):
    X_list, y_list, la_list, components, targets = [], [], [], [], []
    # for _ in meta_data[start_idx: end_idx]:
    for _ in meta_data:
        X_train = _['X_train']
        y_train = _['y_train']
        if downsample:
            if X_train.shape[0] > 100:
                idx = np.random.choice(np.arange(X_train.shape[0]), 100, replace=False)
                X_train = X_train[idx, :]
                y_train = y_train[idx]
            if X_train.shape[1] > 100:
                idx = np.random.choice(np.arange(X_train.shape[1]), 100, replace=False)
                X_train = X_train[:, idx]

        X_list.append(torch.from_numpy(X_train).float())
        y_list.append(torch.from_numpy(y_train).float())
        la_list.append(_['la'])
        components.append(_['components'])
        targets.append(_['performance'])

    la_list = torch.tensor(la_list).unsqueeze(1)
    components = torch.from_numpy(np.stack(components)).float()
    targets = torch.tensor(targets).float()

    return X_list, y_list, la_list, components, targets

class meta_predictor(nn.Module):
    def __init__(self, n_col, n_per_col, embedding_dim=3, meta_embedding_dim=32):
        super(meta_predictor, self).__init__()

        self.f = nn.Sequential(
            nn.Linear(2, meta_embedding_dim),
            nn.ReLU()
        )
        self.g = nn.Sequential(
            nn.Linear(meta_embedding_dim, meta_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(meta_embedding_dim // 2, meta_embedding_dim // 4),
            nn.ReLU()
        )
        self.h = nn.Sequential(
            nn.Linear(meta_embedding_dim // 4, meta_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(meta_embedding_dim // 2, meta_embedding_dim),
            nn.ReLU()
        )

        self.embeddings = nn.ModuleList([nn.Embedding(int(n_per_col[i]), embedding_dim) for i in range(n_col)])
        self.classifier = nn.Sequential(
            nn.Linear(meta_embedding_dim + 1 + n_col * embedding_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid())

    def forward(self, X_list, y_list, nla=None, components=None):
        # meta-feature
        X = torch.stack(X_list).unsqueeze(-1)
        y = torch.stack([y.repeat(X.size(2), 1).T.unsqueeze(-1) for y in y_list])

        f_feature = torch.mean(self.f(torch.cat((X, y), dim=-1)), dim=1)
        g_feature = torch.mean(self.g(f_feature), dim=1)
        meta_features = self.h(g_feature)

        # # meta feature
        # meta_features = []
        # for X, y in zip(X_list, y_list):
        #     assert len(X.size()) == 2 and len(y.size()) == 1
        #     X = X.unsqueeze(2)
        #     y = y.repeat(X.size(1), 1).T.unsqueeze(2)
        #
        #     f_feature = torch.mean(self.f(torch.cat((X, y), dim=2)), dim=0)
        #     g_feature = torch.mean(self.g(f_feature), dim=0)
        #     meta_features.append(self.h(g_feature))
        # # stack the meta-features of multiple datasets
        # meta_features = torch.stack(meta_features)

        # network components' embedding
        assert components.size(1) == len(self.embeddings)

        embedding_list = []
        for i, e in enumerate(self.embeddings):
            embedding_list.append(e(components[:, i].long()))

        embedding = torch.cat(embedding_list, dim=1)
        embedding = torch.cat((meta_features, nla, embedding), dim=1)
        pred = self.classifier(embedding)

        return meta_features, embedding, pred


def fit(meta_data, model, optimizer, batch_size=64):
    criterion = nn.MSELoss()
    epochs = 1

    loss_epoch = []
    for i in tqdm(range(epochs)):
        loss_batch = []
        # for j in range(len(meta_data) // batch_size):
        #     X_list, y_list, la_list, components, targets = dataloader(meta_data, batch_size * j, batch_size * (j + 1))

        for meta_data_batch in tqdm(meta_data):
            X_list, y_list, la_list, components, targets = dataloader(meta_data_batch)

            # clear grad
            model.zero_grad()

            # loss forward
            _, _, pred = model(X_list, y_list, la_list, components)
            loss = criterion(pred.squeeze(), targets)

            # loss backward
            loss.backward()

            # update
            optimizer.step()

            loss_batch.append(loss.item())

        loss_epoch.append(np.mean(loss_batch))

        # print(f'Epoch: {i}--Loss: {np.mean(loss_batch)}')

    # plt.plot(loss_epoch)
    # plt.title('Training Loss')

metric = 'AUCPR'

# current SOTA
result_SOTA_semi = pd.read_csv(os.path.join('result', metric + '_SOTA_semi-supervise.csv'))
result_SOTA_sup = pd.read_csv(os.path.join('result', metric + '_SOTA_supervise.csv'))

assert all(result_SOTA_semi.iloc[:, 0].values == result_SOTA_sup.iloc[:, 0].values)
result_SOTA = pd.concat([result_SOTA_semi, result_SOTA_sup.iloc[:, 1:]], axis=1)
result_SOTA.rename(columns={'Unnamed: 0':'Components'}, inplace=True)
del result_SOTA_semi, result_SOTA_sup

pred_performances = []
for i in tqdm(range(result_SOTA.shape[0])):
    test_dataset, test_la, _ = ast.literal_eval(result_SOTA['Components'].values[i])

# generate training data for meta predictor
meta_data = []
for la in [5, 10, 25, 50]:
    result = pd.read_csv('result/result_' + metric + '_' + str(la) + '_small_500.csv')
    result.rename(columns={'Unnamed: 0':'Components'}, inplace=True)
    # components
    components_df_index = components_process(result)

    # meta data batch
    for i in tqdm(range(1, result.shape[1])):
        current_dataset = result.columns[i]
        if current_dataset == test_dataset:
            continue

        # generate dataset
        data_generator.dataset = current_dataset
        data_generator.seed = 1
        data = data_generator.generator(la=la)

        meta_data_batch = []
        for j in range(result.shape[0]):
            if not pd.isnull(result.iloc[j, i]):  # set nan to 0?
                meta_data_batch.append({'X_train': data['X_train'],
                                        'y_train': data['y_train'],
                                        'dataset_idx': i,
                                        'la': la,
                                        'components': components_df_index.iloc[j, :].values,
                                        'performance': result.iloc[j, i]})
        meta_data.append(meta_data_batch)

# initialization
utils.set_seed(42)
model = meta_predictor(n_col=components_df_index.shape[1],
                       n_per_col=[max(components_df_index.iloc[:, i])+1
                                  for i in range(components_df_index.shape[1])])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# fitting
fit(meta_data, model, optimizer)

# testing
data_generator = DataGenerator(dataset=test_dataset, seed=1)
test_data = data_generator.generator(la=test_la)

with torch.no_grad():
    _, _, pred = model([torch.from_numpy(test_data['X_train']).float() for i in range(components_df_index.shape[0])],
                       [torch.from_numpy(test_data['y_train']).float() for i in range(components_df_index.shape[0])],
                        torch.tensor([test_la for i in range(components_df_index.shape[0])]).unsqueeze(1),
                        torch.from_numpy(components_df_index.values).float())