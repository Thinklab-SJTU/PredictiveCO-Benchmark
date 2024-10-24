import os
import pickle

import numpy as np
import pandas as pd
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem


class Advertising(PTOProblem):
    """ """

    def __init__(self, avg_budget, data_dir, prob_version="real", **kwargs):
        super(Advertising, self).__init__(data_dir)
        self.avg_budget = avg_budget
        self.cost_pv = 0
        if prob_version == "real":
            #
            self.pretrain_X, self.pretrain_Y, self.pretrain_aux = self.load_data(
                f"{data_dir}/train.pickle"
            )
            self.train_X, self.train_Y, self.train_aux = self.load_data(
                f"{data_dir}/train_mock.pickle", isMock=True
            )
            self.test_X, self.test_Y, self.test_aux = self.load_data(
                f"{data_dir}/test_mock.pickle", isMock=True
            )
        else:
            assert False, "Not implemented"

    def load_data(self, path, isMock=False):
        return 
    
    def get_pretrain_data(self, **kwargs):
        return self.pretrain_X, self.pretrain_Y, self.pretrain_aux

    def get_train_data(self, **kwargs):
        return self.train_X, self.train_Y, self.train_aux

    def get_val_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_test_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_model_shape(self):
        return self.train_X[0].shape[-1], 1

    def get_output_activation(self):
        return "sigmoid"

    def get_twostageloss(self):
        return "bce"

    def is_eval_train(self):
        return False

    def get_decision(self, Y, params, ptoSolver, isTrain=True, **kwargs):
        sols, objs = 0, 0
        return sols, objs

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        objs = 0
        return objs

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "avg_budget": self.avg_budget,
        }

    def get_eval_metric(self):
        return "uplift"




