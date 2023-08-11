import torch
from torch import nn

# Using Encoder-Decoder model structure for pretraining
class Pretrained_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Pretrained_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        # ModuleList
        for e in self.encoder:
            X = e(X)

        for d in self.decoder:
            X = d(X)
        return X

class Pretrained_Model_ResNet(nn.Module):
    def __init__(self, input_size, model):
        super(Pretrained_Model_ResNet, self).__init__()
        self.encoder_decoder = nn.Sequential(
                model.first_layer,
                model.blocks,
                nn.Linear(128, input_size)
            )

    def forward(self, X):
        X = self.encoder_decoder(X)
        return X

class Pretrained_Model_FTT(nn.Module):
    def __init__(self, input_size, model):
        super(Pretrained_Model_FTT, self).__init__()
        self.encoder_decoder = nn.Sequential(
                model.feature_tokenizer,
                model.cls_token,
                model.transformer,
                nn.Linear(8, input_size)
            )

    def forward(self, X):
        X = self.encoder_decoder(X)
        return X

# MLP backbone
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

# MLP backbone for paired input (e.g., PReNet)
class MLP_pair(nn.Module):
    def __init__(self, layers, input_size, hidden_size_list, act_fun, p):
        super(MLP_pair, self).__init__()
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
        self.reg = nn.Linear(20*2, 1)

    def forward(self, X_left, X_right):
        # feature representation
        for i, f in enumerate(self.feature):
            X_left = f(X_left)
            X_right = f(X_right)

        # anomaly scoring
        score = self.reg(torch.cat((X_left, X_right), dim=1))

        return score

# AutoEncoder backbone, from "Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection"
class AE(nn.Module):
    def __init__(self, layers, input_size, hidden_size_list, act_fun, p):
        super(AE, self).__init__()
        assert layers == len(hidden_size_list)

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