import os

import numpy as np
import torch


# from decorators import input_to_numpy
# from utils import TrainingIterator
from gurobipy import GRB  # pylint: disable=no-name-in-module
from torchvision import transforms as transforms

from openpto.method.utils_method import to_array, to_device, to_tensor
from openpto.problems.PTOProblem import PTOProblem


class Shortestpath(PTOProblem):
    """ """

    def __init__(
        self,
        num_train_instances=400,  # number of instances to use from the dataset to train
        num_test_instances=200,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for validation
        size=12,
        normalize=True,
        rand_seed=0,  # for reproducibility
        prob_version="warcraft",
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(Shortestpath, self).__init__()
        self._set_seed(rand_seed)
        self.size = size
        self.normalize = normalize
        self.prob_version = prob_version
        # split
        self.n_vals = int(num_train_instances * val_frac)
        self.n_trains = num_train_instances - self.n_vals
        self.n_tests = num_test_instances
        ###
        if prob_version == "warcraft":
            self.load_dataset(
                data_dir + f"/{size}x{size}/",
                normalize=normalize,
            )
        else:
            raise NotImplementedError

    @staticmethod
    def do_norm(inputs):
        in_mean, in_std = (
            torch.mean(inputs, axis=(0, 1, 2), keepdims=True),
            torch.std(inputs, axis=(0, 1, 2), keepdims=True),
        )
        return (inputs - in_mean) / in_std

    def read_data(self, data_dir, split_prefix, normalize):
        return

    def load_dataset(self, data_dir, normalize):
        return


    def get_train_data(self, train_mode="iid", **kwargs):
        return self.train_X, self.train_Y, self.train_Z

    def get_val_data(self, train_mode="iid", **kwargs):
        return self.val_X, self.val_Y, self.val_Z

    def get_test_data(self, train_mode="iid", **kwargs):
        return self.test_X, self.test_Y, self.test_Z

    def get_model_shape(self):
        assert self.train_X.shape[2] == 8 * self.size
        return self.train_X.shape[2], self.size**2

    def get_eval_metric(self):
        return "regret"

    def get_output_activation(self):
        if self.prob_version == "direct":
            return "sigmoid"
        else:
            return "identity"

    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        sol, obj = 0, 0
        return sol, obj

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        obj = 0
        return obj

    def get_twostageloss(self):
        if self.prob_version == "direct":
            return "bce"
        else:
            return "mse"

    def init_API(self):
        return {
            "modelSense": GRB.MINIMIZE,
            "n_vars": self.size**2,
            "size": self.size,
        }

