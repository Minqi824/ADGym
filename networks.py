import torch
from torch import nn
import numpy as np

# ADSD based on MLP backbone
class MLP(nn.Module):
    def __init__(self, layers, input_size, hidden_size_list, act_fun, p):
        super(MLP, self).__init__()
        assert layers == len(hidden_size_list)

        # feature representation layer
        self.feature = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.feature.append(nn.Sequential(nn.Linear(input_size, hidden_size_list[i]),
                                                  act_fun,
                                                  nn.Dropout(p=p)))
            else:
                self.feature.append(nn.Sequential(nn.Linear(hidden_size_list[i-1], hidden_size_list[i]),
                                                  act_fun,
                                                  nn.Dropout(p=p)))

        # anomaly scoring layer
        self.reg = nn.Linear(20, 1)

    def forward(self, X):
        # feature representation
        for f in self.feature:
            X = f(X)

        # anomaly scoring
        score = self.reg(X)

        return score

# ADSD based on AutoEncoder backbone, from "Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection"
class AE(nn.Module):
    def __init__(self, layers, input_size, hidden_size_list, act_fun, p):
        super(AE, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(layers):
            if i == 0:
                self.encoder.append(nn.Sequential(nn.Linear(input_size, hidden_size_list[i]),
                                                  act_fun,
                                                  nn.Dropout(p=p)))
            else:
                self.encoder.append(nn.Sequential(nn.Linear(hidden_size_list[i-1], hidden_size_list[i]),
                                                  act_fun,
                                                  nn.Dropout(p=p)))

        for i in range(layers-1, -1, -1):
            if i == 0:
                self.decoder.append(nn.Sequential(nn.Linear(hidden_size_list[i], input_size),
                                                  act_fun,
                                                  nn.Dropout(p=p)))
            else:
                self.decoder.append(nn.Sequential(nn.Linear(hidden_size_list[i], hidden_size_list[i-1]),
                                                  act_fun,
                                                  nn.Dropout(p=p)))

        self.reg_1 = nn.Sequential(
            nn.Linear(input_size+hidden_size_list[-1]+1, 256),
            act_fun,
            nn.Dropout(p=p)
        )

        self.reg_2 = nn.Sequential(
            nn.Linear(256+1, 32),
            act_fun,
            nn.Dropout(p=p)
        )

        self.reg_3 = nn.Sequential(
            nn.Linear(32+1, 1)
        )

    def forward(self, X):
        # hidden representation
        for i, e in enumerate(self.encoder):
            if i == 0:
                h = e(X)
            else:
                h = e(h)

        # reconstructed input vector
        for i, d in enumerate(self.decoder):
            if i == 0:
                X_hat = d(h)
            else:
                X_hat = d(X_hat)

        # reconstruction residual vector
        r = torch.sub(X_hat, X)

        # reconstruction error
        e = r.norm(dim=1).reshape(-1, 1)

        # normalized reconstruction residual vector
        r = torch.div(r, e) #div by broadcast

        # regression
        feature = self.reg_1(torch.cat((h, r, e), dim=1))
        feature = self.reg_2(torch.cat((feature, e), dim=1))
        score = self.reg_3(torch.cat((feature, e), dim=1))

        return score