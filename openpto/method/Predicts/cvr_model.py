import torch
import torch.nn as nn


class CVRModel(nn.Module):
    def __init__(self, **args):
        super(CVRModel, self).__init__()
        # parameters
        # n_users = args['n_users']
        input_dim = args["input_dim"]
        hidden_dim = args["emb_dim"]
        # models
        self.user_emb_layer = nn.Embedding(4, hidden_dim)
        self.pred0 = nn.Linear(input_dim, hidden_dim)
        self.pred1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.pred2 = nn.Linear(hidden_dim * 4, hidden_dim // 2)
        self.pred3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, input_feat):
        # user_feat = self.user_emb_layer(u_node)
        learned_feat = self.pred0(input_feat)
        # user_feat = self.relu(self.down(torch.cat((user_feat,stats_feat),-1)))
        learned_feat = self.relu(learned_feat)
        learned_feat = self.relu(self.pred1(learned_feat))
        learned_feat = self.relu(self.pred2(learned_feat))
        logits = self.pred3(learned_feat)
        return torch.sigmoid(logits)
