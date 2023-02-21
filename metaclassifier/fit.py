import torch
from torch import nn
import numpy as np
from tqdm import tqdm


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

def fit_end2end(meta_data, model, optimizer, batch_size=64):
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