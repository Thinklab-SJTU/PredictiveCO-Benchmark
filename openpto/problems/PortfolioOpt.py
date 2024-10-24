import datetime as dt
import os
import random

import pandas as pd
import quandl
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem


class PortfolioOpt(PTOProblem):
    """ """

    def __init__(
        self,
        num_train_instances=200,  # number of *days* to use from the dataset to train
        num_test_instances=200,  # number of *days* to use from the dataset to test
        num_stocks=50,  # number of stocks per instance to choose from
        val_frac=0.2,  # fraction of training data reserved for test
        rand_seed=0,  # for reproducibility
        alpha=0.1,  # risk aversion constant
        prob_version="real",
        data_dir="openpto/data",  # directory to store data
    ):
        super(PortfolioOpt, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)

        # Load train and test labels
        self.num_stocks = num_stocks
        self.Xs, self.Ys, self.covar_mat = self._load_instances(data_dir, num_stocks)
        # Split data into train/val/test
        total_days = self.Xs.shape[0]
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        num_days = self.num_train_instances + self.num_test_instances
        assert self.num_train_instances + self.num_test_instances < total_days
        assert 0 < val_frac < 1
        self.val_frac = val_frac

        #   Creating "days" for train/valid/test
        idxs = list(range(num_days))
        num_val = int(self.val_frac * self.num_train_instances)
        self.train_idxs = idxs[: self.num_train_instances - num_val]
        self.val_idxs = idxs[
            self.num_train_instances - num_val : self.num_train_instances
        ]
        self.test_idxs = idxs[self.num_train_instances :]
        assert all(
            x is not None for x in [self.train_idxs, self.val_idxs, self.test_idxs]
        )

        # Create functions for optimisation
        self.alpha = alpha

        # Undo random seed setting
        self._set_seed()

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "alpha": self.alpha,
            "num_stocks": self.num_stocks,
        }

    def _load_instances(
        self,
        data_dir,
        stocks_per_instance,
        reg=0.1,
    ):
        return

    def _get_price_feature_df(
        self,
        overwrite=False,
    ):
        return

    def _compute_monthly_cols(self, symbol_df):
        return

    def _download_symbols(self):
        return

    def _download_prices(self, symbol_df):
        return

    def _load_raw_symbols(
        self,
        overwrite=False,
    ):
       return

    def _get_price_feature_matrix(self, price_feature_df):
        return 

    def _get_data(
        self,
        data_dir,
        start_date=dt.datetime(2004, 1, 1),
        end_date=dt.datetime(2017, 1, 1),
        collapse="daily",
        overwrite=False,
    ):
        return 

    def get_train_data(self, **kwargs):
        return (
            self.Xs[self.train_idxs],
            self.Ys[self.train_idxs],
            self.covar_mat[self.train_idxs],
        )

    def get_val_data(self, **kwargs):
        return (
            self.Xs[self.val_idxs],
            self.Ys[self.val_idxs],
            self.covar_mat[self.val_idxs],
        )

    def get_test_data(self, **kwargs):
        return (
            self.Xs[self.test_idxs],
            self.Ys[self.test_idxs],
            self.covar_mat[self.test_idxs],
        )

    def get_model_shape(self):
        return self.Xs.shape[-1], 1

    def get_twostageloss(self):
        return "mse"

    def _get_covar_mat(self, instance_idxs):
        return 

    def get_decision(
        self, Y, aux_data=None, ptoSolver=None, max_instances_per_batch=1500, **kwargs
    ):
        sols, objs = 0, 0
        return sols, objs


    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        obj = 0
        return obj

    def get_output_activation(self):
        return "tanh"


if __name__ == "__main__":
    problem = PortfolioOpt()
    X_train, Y_train, Y_train_aux = problem.get_train_data()

    Z_train = problem.get_decision(Y_train, aux_data=Y_train_aux)
    obj = problem.get_objective(Y_train, Z_train, aux_data=Y_train_aux)
