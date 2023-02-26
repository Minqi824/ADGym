import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from data_generator import DataGenerator


def fit(train_loader, model, optimizer, epochs=20):
    criterion = nn.MSELoss()

    loss_epoch = []
    for i in range(epochs):
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
        print(f'Epoch: {i}--Loss: {round(np.mean(loss_batch), 4)}')

def fit_end2end(meta_data, model, optimizer, epochs=5):
    criterion = nn.MSELoss()

    loss_epoch = []
    for i in range(epochs):
        loss_batch = []
        for meta_data_batch in meta_data:
            X_list, y_list, la_list, components, targets = meta_data_batch

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
        print(f'Epoch: {i}--Loss: {round(np.mean(loss_batch), 4)}')