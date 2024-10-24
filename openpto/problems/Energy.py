import os

import numpy as np
import pandas as pd
import sklearn
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from openpto.method.Solvers.grb.grb_energy import ICONGrbSolver
from openpto.problems.PTOProblem import PTOProblem


class Energy(PTOProblem):
    """ """

    def __init__(
        self,
        prob_version="energy",
        num_train_instances=0,
        num_test_instances=0,
        rand_seed=0,
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(Energy, self).__init__(data_dir)
        self.prob_version = prob_version
        self._set_seed(rand_seed)
        self.rand_seed = rand_seed
        # Obtain data
        if prob_version == "energy":
            self.get_energy_data()

    def get_twostageloss(self):
        return "mse"

    def get_energy_data(self):
        return

    def get_train_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.train_idxs],
            self.Ys[self.train_idxs],
            self.Ys[self.train_idxs],  # placeholder not used
        )

    def get_val_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.val_idxs],
            self.Ys[self.val_idxs],
            self.Ys[self.val_idxs],  # placeholder not used
        )

    def get_test_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.test_idxs],
            self.Ys[self.test_idxs],
            self.Ys[self.test_idxs],  # placeholder not used
        )

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        obj = 0
        return obj

    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        sols, objs = 0, 0
        return sols, objs

    def init_API(self):
        dirct = f"{self.data_dir}/SchedulingInstances"
        os.listdir(dirct)[0]
        reading_dict = self.problem_data_reading(
            f"{self.data_dir}/SchedulingInstances/load1/day01.txt"
        )
        out_dict = {**reading_dict, **{"modelSense": GRB.MINIMIZE}}
        return out_dict

    def get_model_shape(self):
        if self.prob_version == "gen":
            return self.Xs[self.train_idxs].shape[-1], self.num_items
        else:
            return self.Xs[self.train_idxs].shape[-1], 1

    def get_output_activation(self):
        return "identity"

    # prep numpy arrays, Xs will contain groupID as first column
    def get_energy(self, fname=None, trainTestRatio=0.70):
        return

    def get_energy_grouped(self, fname=None):
        return

    def get_energy_pandas(self, fname=None):
        return 

    def problem_data_reading(self, filename):
        return
