import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from data_generator import DataGenerator
from utils import Utils

utils = Utils()

def fit(train_loader, model, optimizer, epochs, val_loader=None, es=False, tol:int = 5):
    best_metric = -9999; t = 0

    loss_epoch = []
    for i in range(epochs):
        loss_batch = []
        for batch in train_loader:
            batch_meta_features, batch_la, batch_components, batch_y = batch

            # clear grad
            model.zero_grad()

            # loss forward
            _, pred = model(batch_meta_features, batch_la.unsqueeze(1), batch_components)
            loss = 1 - utils.criterion(y_pred=pred.squeeze(), y_true=batch_y)

            # loss backward
            loss.backward()

            # update
            optimizer.step()
            loss_batch.append(loss.item())

        loss_epoch.append(np.mean(loss_batch))

        if val_loader is not None and es:
            val_metric = utils.evaluate(model, val_loader=val_loader, device=batch_y.device)
            print(f'Epoch: {i}--Training Loss: {round(np.mean(loss_batch), 4)}---Validation Metric: {round(val_metric.item(), 4)}')
            if val_metric > best_metric:
                best_metric = val_metric
                t = 0
            else:
                t += 1

            if t > tol:
                print(f'Early stopping at epoch: {i}!')
                break
        else:
            print(f'Epoch: {i}--Loss: {round(np.mean(loss_batch), 4)}')

    return i

def fit_end2end(meta_data, model, optimizer, epochs=10):
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