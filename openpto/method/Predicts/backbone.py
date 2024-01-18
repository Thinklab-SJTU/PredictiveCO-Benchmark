import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        num_layers,
        intermediate_size,
        output_activation,
        **args,
    ):
        super(Resnet, self).__init__()
        # models
        # self.user_emb_layer = nn.Embedding(4, hidden_dim)
        self.pred0 = nn.Linear(num_features, intermediate_size * 4)
        self.pred1 = nn.Linear(intermediate_size * 4, intermediate_size * 2)
        self.pred2 = nn.Linear(intermediate_size * 2, intermediate_size // 2)
        self.pred3 = nn.Linear(intermediate_size // 2, num_targets)

    def forward(self, input_feat):
        out = input_feat
        Loss = list()
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)
        return out
