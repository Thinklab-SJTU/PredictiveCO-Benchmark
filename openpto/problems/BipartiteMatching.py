import pickle
import random

import networkx as nx
import numpy as np
import torch

from gurobipy import GRB

from openpto.method.Solvers.cvxpy.cp_bmatching import BmatchingSolver
from openpto.method.utils_method import to_array, to_tensor
from openpto.problems.PTOProblem import PTOProblem


class BipartiteMatching(PTOProblem):
    """ """

    def __init__(
        self,
        num_train_instances=20,  # number of instances to use from the dataset to train
        num_test_instances=6,  # number of instances to use from the dataset to test
        num_nodes=50,  # number of nodes in the LHS and RHS of the bipartite matching graphs
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        prob_version="cora",
        data_dir="./openpto/data/",
    ):
        super(BipartiteMatching, self).__init__(data_dir)
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        # Load train and test labels
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.num_nodes = num_nodes
        self.Xs, self.Ys = self._load_instances(
            self.num_train_instances, self.num_test_instances, self.num_nodes
        )
        # Split data into train/val/test
        assert 0 < val_frac < 1
        self.val_frac = val_frac

        idxs = list(range(self.num_train_instances + self.num_test_instances))
        random.shuffle(idxs)
        self.val_idxs = idxs[0 : int(self.val_frac * self.num_train_instances)]
        self.train_idxs = idxs[
            int(self.val_frac * self.num_train_instances) : self.num_train_instances
        ]
        self.test_idxs = idxs[self.num_train_instances :]
        assert all(
            x is not None for x in [self.train_idxs, self.val_idxs, self.test_idxs]
        )

        # Create functions for optimisation
        self.opt_train = BmatchingSolver()._getModel(
            isTrain=True, num_nodes=self.num_nodes
        )
        self.opt_test = BmatchingSolver()._getModel(
            isTrain=False, num_nodes=self.num_nodes
        )

        # Undo random seed setting
        self._set_seed()

    def _load_instances(
        self,
        num_train_instances,
        num_test_instances,
        num_nodes,
        random_split=True,
        verbose=False,
    ):
        return

    def get_train_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.train_idxs],
            self.Ys[self.train_idxs],
            self.Ys[self.train_idxs],
        )

    def get_val_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.val_idxs],
            self.Ys[self.val_idxs],
            self.Ys[self.val_idxs],
        )

    def get_test_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.test_idxs],
            self.Ys[self.test_idxs],
            self.Ys[self.test_idxs],
        )

    def get_model_shape(self):
        return self.Xs.shape[-1], 1

    def get_output_activation(self):
        return "sigmoid"

    def get_twostageloss(self):
        return "bce"

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        objs = 0
        return objs

    def get_decision(
        self,
        Y,
        params,
        ptoSolver=None,
        isTrain=False,
        max_instances_per_batch=5000,
        **kwargs,
    ):
        sols, objs = 0, 0
        return sols, objs

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
        }


if __name__ == "__main__":
    problem = BipartiteMatching()
