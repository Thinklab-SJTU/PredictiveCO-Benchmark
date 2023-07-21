from numpy import square
from math import sqrt
from functools import reduce
import operator

import torch

from openpto.method.Solvers.utils_solver import View


######################## prediction model wrapper  ############################
def pred_model_wrapper_solver(args):
    model_dict = {"dense": dense_nn}
    # TODO:more pred models
    return model_dict[args.pred_model]


#################################### Dense NN #################################
# TODO: Pretty it up
def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation="relu",
    output_activation="sigmoid",
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "sigmoid":
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception("Invalid activation function: " + str(activation))
        net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(
                torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1))
            )
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [torch.nn.Linear(num_features, num_targets)]
        else:
            net_layers = [
                torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)),
                View(num_targets),
            ]

    if output_activation == "relu":
        net_layers.append(torch.nn.ReLU())
    elif output_activation == "sigmoid":
        net_layers.append(torch.nn.Sigmoid())
    elif output_activation == "tanh":
        net_layers.append(torch.nn.Tanh())
    elif output_activation == "softmax":
        net_layers.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*net_layers)
