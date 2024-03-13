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
        input_dim = 3 * num_features**2
        self.fc1 = nn.Linear(in_features=input_dim, out_features=intermediate_size)
        self.fc2 = nn.Linear(in_features=intermediate_size, out_features=num_targets)
        self.act_func = act_dict[activation]
        self.out_act_func = act_dict[output_activation]

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.act_func(self.fc1(x))
        x = self.out_act_func(self.fc2(x))
        return x


class Resnet18(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        output_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(
            pretrained=False, num_classes=num_targets
        )
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(
            num_features, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.out_act_func = act_dict[output_activation]

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        # print("x: ", x.shape)  # x:  torch.Size([1000, 96, 96, 3])
        x = self.resnet_model(x)  # torch.Size([1000, 144])
        x = self.out_act_func(x)
        return x


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
        self.resnet_model = torchvision.models.resnet18(
            pretrained=False, num_classes=num_targets
        )
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(
            num_features, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        output_shape = (int(sqrt(num_targets)), int(sqrt(num_targets)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        # self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)
        self.out_act_func = act_dict[output_activation]

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        # x = self.resnet_model.layer2(x)
        # x = self.resnet_model.layer3(x)
        # x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        x = x.reshape(x.shape[0], -1)
        x = self.out_act_func(x)
        return x


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
        self.conv1 = nn.Conv2d(
            num_features=num_features,
            out_channels=intermediate_size,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            num_features=intermediate_size,
            out_channels=intermediate_size,
            kernel_size=kernel_size,
            stride=stride,
        )

        output_shape = (4, 4)
        self.pool = nn.AdaptiveAvgPool2d(output_shape)

        self.fc1 = nn.Linear(
            in_features=output_shape[0] * output_shape[1] * intermediate_size,
            num_targets=linear_layer_size,
        )
        self.fc2 = nn.Linear(in_features=linear_layer_size, num_targets=num_targets)
        self.out_act_func = act_dict[output_activation]

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.out_act_func(x)
        return x


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
        # self.use_second_conv = use_second_conv

        self.conv1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=intermediate_size,
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=intermediate_size,
            out_channels=intermediate_size,
            kernel_size=kernel_size,
            stride=1,
        )

        output_shape = (int(sqrt(num_targets)), int(sqrt(num_targets)))
        if pooling == "average":
            self.pool = nn.AdaptiveAvgPool2d(output_shape)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_shape)

        self.conv3 = nn.Conv2d(
            in_channels=intermediate_size, out_channels=1, kernel_size=1, stride=1
        )
        self.act_func = act_dict[activation]
        self.out_act_func = act_dict[output_activation]

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.act_func(self.conv1(x))
        x = self.act_func(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.out_act_func(x)
        x = x.reshape(x.shape[0], -1)
        return x
