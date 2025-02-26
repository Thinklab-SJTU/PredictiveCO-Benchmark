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
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softmax": partial(nn.Softmax, dim=-1),
    "sigmoid": nn.Sigmoid(),
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
        if num_layers > 1:
            if intermediate_size is None:
                intermediate_size = max(num_features, num_targets)
            if activation in ["relu", "sigmoid", "tanh"]:
                activation_fn = act_func_dict[activation]
            else:
                raise Exception("Invalid activation function: " + str(activation))
            net_layers = [
                nn.Linear(num_features, intermediate_size),
                activation_fn,
            ]
            for _ in range(num_layers - 2):
                net_layers.append(nn.Linear(intermediate_size, intermediate_size))
                net_layers.append(activation_fn)
            if not isinstance(num_targets, tuple):
                net_layers.append(nn.Linear(intermediate_size, num_targets))
            else:
                net_layers.append(
                    nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1))
                )
                net_layers.append(View(num_targets))
        else:
            if not isinstance(num_targets, tuple):
                net_layers = [nn.Linear(num_features, num_targets)]
            else:
                net_layers = [
                    nn.Linear(num_features, reduce(operator.mul, num_targets, 1)),
                    View(num_targets),
                ]
        if output_activation not in ["identity", "none"]:
            net_layers.append(act_func_dict[output_activation])

        self.net = nn.Sequential(*net_layers)

    def forward(self, X):
        return self.net(X)


#################################### Dense NN #################################
# function version of MLP
def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=32,
    activation="relu",
    output_activation="sigmoid",
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation in ["relu", "sigmoid", "tanh"]:
            activation_fn = act_func_dict[activation]
        else:
            raise Exception("Invalid activation function: " + str(activation))
        net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(
                nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1))
            )
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [nn.Linear(num_features, num_targets)]
        else:
            net_layers = [
                nn.Linear(num_features, reduce(operator.mul, num_targets, 1)),
                View(num_targets),
            ]
    net_layers.append(act_func_dict[output_activation])

    return nn.Sequential(*net_layers)
