# training a meta classifier to predict detection performance by given:
# 1. meta-feature (either use end-to-end training strategy, e.g., dataset2vec or use metaod to extract)
# 2. number of labeled anomalies
# 3. network components

# To Do
# compare to current SOTA
# number of labeled anomalies
# end-to-end meta-feature (should we follow a pretrained-finetune)

from torch import nn
import torch


class meta_feature_extractor(nn.Module):
    def __init__(self):
        super(meta_feature_extractor, self).__init__()

        self.f = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.g = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.h = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 200)
        )

    def forward(self, X, y):
        assert len(X.size()) == 2 and len(y.size()) == 1
        X = X.unsqueeze(2)
        y = y.repeat(X.size(1), 1).T.unsqueeze(2)

        f_feature = torch.mean(self.f(torch.cat((X, y), dim=2)), dim=0)
        g_feature = torch.mean(self.g(f_feature), dim=0)
        h_feature = self.h(g_feature)

        return h_feature

class meta_predictor(nn.Module):
    def __init__(self, n_col, n_per_col, embedding_dim=3):
        super(meta_predictor, self).__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(int(n_per_col[i]), embedding_dim) for i in range(n_col)])
        self.classifier = nn.Sequential(
            nn.Linear(200 + 1 + n_col * embedding_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid())

    def forward(self, meta_features=None, nla=None, components=None):
        assert components.size(1) == len(self.embeddings)

        embedding_list = []
        for i, e in enumerate(self.embeddings):
            embedding_list.append(e(components[:, i].long()))

        embedding = torch.cat(embedding_list, dim=1)
        embedding = torch.cat((meta_features, nla, embedding), dim=1)
        pred = self.classifier(embedding)

        return embedding, pred