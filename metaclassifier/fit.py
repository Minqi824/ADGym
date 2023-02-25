import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from data_generator import DataGenerator


def fit(train_loader, model, optimizer, epochs: int=20):
    criterion = nn.MSELoss()

    loss_epoch = []
    for i in tqdm(range(epochs)):
        loss_batch = []
        for batch in train_loader:
            batch_meta_features, batch_la, batch_components, batch_y = batch

            # clear grad
            model.zero_grad()

            # loss forward
            _, pred = model(batch_meta_features, batch_la.unsqueeze(1), batch_components)
            loss = criterion(pred.squeeze(), batch_y)

            # loss backward
            loss.backward()

            # update
            optimizer.step()

            loss_batch.append(loss.item())

        loss_epoch.append(np.mean(loss_batch))
        print(f'Epoch: {i}--Loss: {np.mean(loss_batch)}')

class fit_end2end():
    def __init__(self, seed):
        self.seed = seed
        self.data_generator = DataGenerator()

    # dataloader for end2end meta classifier version
    def dataloader(self, meta_data, downsample=True):
        X_list, y_list, la_list, components, targets = [], [], [], [], []
        for _ in meta_data:
            X_train = _['X_train']
            y_train = _['y_train']
            if downsample:
                if X_train.shape[0] > 1000:
                    idx = np.random.choice(np.arange(X_train.shape[0]), 1000, replace=False)
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

    def fit(self, meta_data, model, optimizer, epochs=5):
        criterion = nn.MSELoss()

        loss_epoch = []
        for i in tqdm(range(epochs)):
            loss_batch = []
            for meta_data_batch in meta_data:
                X_list, y_list, la_list, components, targets = self.dataloader(meta_data_batch)

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
            print(f'Epoch: {i}--Loss: {np.mean(loss_batch)}')