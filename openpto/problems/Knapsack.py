import numpy as np
import pandas as pd
import sklearn
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module
from sklearn.preprocessing import StandardScaler

from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem


class Knapsack(PTOProblem):
    """
    Knapsack problem
    """

    def __init__(
        self,
        num_train_instances=400,  # number of instances to use from the dataset to train
        num_test_instances=200,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=2023,  # for reproducibility
        prob_version="gen",
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(Knapsack, self).__init__(data_dir)
        self.kwargs = kwargs
        self.rand_seed = rand_seed
        self.prob_version = prob_version
        if self.prob_version in ["gen", "gen-ood"]:
            num_items = kwargs["num_items"]
            knapsack_dim, num_features = kwargs["knapsack_dim"], kwargs["num_features"]
            mean, var = kwargs["mean"], kwargs["var"]
            poly_deg, noise_width = kwargs["poly_deg"], kwargs["noise_width"]
            distr = kwargs["distr"]
        if "envs" in kwargs:
            self.env_config = kwargs["envs"]
            print("self.env_config: ", self.env_config)
        self.capacity = kwargs["capacity"]
        self.prob_version = prob_version
        self.rand_seed = rand_seed
        self._set_seed(rand_seed)
        ### Obtain data
        if prob_version == "energy":
            self.get_energy_data(val_frac)
        elif prob_version == "gen":
            self.num_items = num_items
            n_vals = int(val_frac * num_train_instances)
            n_trains = num_train_instances - n_vals
            ### gen data
            print("distribution of generated data, mean, var:", mean, var)
            weights, feats, profits = self.genKPData(
                num_train_instances + num_test_instances,
                num_features,
                num_items,
                mean,
                var,
                dim=knapsack_dim,
                poly_deg=poly_deg,
                noise_width=noise_width,
                distr=distr,
                seed=rand_seed,
            )
            # splits
            assert 0 < val_frac < 1
            self.train_idxs = range(n_vals, num_train_instances)
            self.val_idxs = range(n_vals)
            self.weights = weights
            ### train
            self.Xs_train, self.Ys_train = (
                feats[self.train_idxs],
                profits[self.train_idxs],
            )  # (bz, feature_dim), (bz, n_items)
            self.params_train = weights.unsqueeze(0).expand(n_trains, -1)
            ### val
            self.Xs_val, self.Ys_val = (
                feats[self.val_idxs],
                profits[self.val_idxs],
            )
            self.params_val = weights.unsqueeze(0).expand(n_vals, -1)
            ### test
            self.Xs_test, self.Ys_test = (
                feats[num_train_instances:],
                profits[num_train_instances:],
            )
            self.params_test = weights.unsqueeze(0).expand(num_test_instances, -1)
            ### Done
        else:
            raise ValueError("Not a valid problem version: {}".format(prob_version))

    def get_train_data(self, train_mode="iid", **kwargs):
        if train_mode == "iid":
            return (
                self.Xs_train,
                self.Ys_train,
                self.params_train,
            )
        else:
            raise NotImplementedError

    def get_val_data(self, train_mode="iid", **kwargs):
        if train_mode == "iid":
            return (
                self.Xs_val,
                self.Ys_val,
                self.params_val,
            )
        else:
            raise NotImplementedError

    def get_test_data(self, train_mode="iid", **kwargs):
        return self.Xs_test, self.Ys_test, self.params_test

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        obj = 0
        return obj


    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        sols_array, objs_array = 0, 0
        return sols_array, objs_array

    def init_API(self):
        return {
            "weights": self.weights,
            "capacity": self.capacity,
            "modelSense": GRB.MAXIMIZE,
            "n_items": self.num_items,
            "tau": 1,
        }

    def get_model_shape(self):
        if self.prob_version in ["gen", "gen-ood"]:
            return self.Xs_train.shape[-1], self.num_items
        else:
            return self.Xs_train.shape[-1], 1

    def get_output_activation(self):
        return "identity"

    def get_energy_data(self, val_frac):
        return

    def get_energy(self, fname=None, trainTestRatio=0.70):
        return

    @staticmethod
    def genKPData(
        num_instances,
        num_features,
        num_items,
        mean=0,
        var=1,
        dim=1,
        poly_deg=1,
        noise_width=0,
        distr="normal",
        seed=2023,
    ):
        return

    def get_twostageloss(self):
        return "mse"
