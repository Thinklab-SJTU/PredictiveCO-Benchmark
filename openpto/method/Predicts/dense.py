import operator

from functools import partial, reduce

import torch.nn as nn
import torch.nn.functional as F

from openpto.method.Solvers.utils_solver import View

act_dict = {
    "relu": F.relu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "softplus": F.softplus,
    "softmax": partial(F.softmax, dim=-1),
    "identity": lambda x: x,
}

act_func_dict = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "identity": lambda x: x,
}


class MLP(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        num_layers,
        intermediate_size=32,
        activation="relu",
        output_activation="sigmoid",
        **args,
    ):
        super(MLP, self).__init__()

    def forward(self, X):
        return
