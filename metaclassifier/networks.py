import torch
from torch import nn

# meta classifier by using MetaOD feature extractor
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

# meta classifier via end2end learning
class meta_predictor_end2end(nn.Module):
    def __init__(self, n_col, n_per_col, embedding_dim=3, meta_embedding_dim=32):
        super(meta_predictor_end2end, self).__init__()

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
