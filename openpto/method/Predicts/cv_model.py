"""
    Code adopted from: https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py
"""
from functools import partial
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

act_dict = {
    "relu": F.relu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "softplus": F.softplus,
    "softmax": partial(F.softmax, dim=-1),
    "identity": lambda x: x,
}


class cv_mlp(torch.nn.Module):
    # TODO: debug the training pipeline of image input
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
        super().__init__()

    def forward(self, x):
        return


class Resnet18(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        output_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()

    def forward(self, x):
        return


class CombResnet18(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        activation="relu",
        output_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()


    def forward(self, x):
        return


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        kernel_size,
        stride,
        linear_layer_size,
        intermediate_size=32,
        activation="relu",
        output_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()


    def forward(self, x):
        return


class PureConvNet(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        pooling="average",
        kernel_size=1,
        intermediate_size=32,
        activation="relu",
        output_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()

    def forward(self, x):
        return
