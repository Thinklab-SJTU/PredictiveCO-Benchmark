import pickle
import random

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem


class BudgetAllocation(PTOProblem):
    """ """

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_targets=10,  # number of targets to consider
        num_items=5,  # number of items to choose from
        budget=1,  # number of items that can be picked
        num_fake_targets=20,  # number of random features added to make the task harder
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        prob_version="real",
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(BudgetAllocation, self).__init__(data_dir)
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        train_seed, test_seed = random.randrange(2**32), random.randrange(2**32)

        if prob_version == "real":
            # Load train and test labels
            self.num_train_instances = num_train_instances
            self.num_test_instances = num_test_instances
            Ys_train_test = []
            for seed, num_instances in zip(
                [train_seed, test_seed], [num_train_instances, num_test_instances]
            ):
                # Set seed for reproducibility
                self._set_seed(seed)

                # Load the relevant data (Ys)
                Ys = self._load_instances(num_instances, num_items, num_targets)  # labels
                assert not torch.isnan(Ys).any()

                # Save Xs and Ys
                Ys_train_test.append(Ys)
            self.Ys_train, self.Ys_test = (*Ys_train_test,)

            # Generate features based on the labels
            self.num_targets = num_targets
            self.num_fake_targets = num_fake_targets
            self.num_features = self.num_targets + self.num_fake_targets
            self.Xs_train, self.Xs_test = self._generate_features(
                [self.Ys_train, self.Ys_test]
            )
            # X_train:[400, 5, 10])     Ys_train: [400, 5, 10])  Z: torch.Size([5])
            # assert Z.ndim + 1 == Y.ndim
            assert not (
                torch.isnan(self.Xs_train).any() or torch.isnan(self.Xs_test).any()
            )

            # Split training data into train/val
            assert 0 < val_frac < 1
            self.val_frac = val_frac
            self.val_idxs = range(0, int(self.val_frac * num_train_instances))
            self.train_idxs = range(
                int(self.val_frac * num_train_instances), num_train_instances
            )
            assert all(x is not None for x in [self.train_idxs, self.val_idxs])
        else:
            raise NotImplementedError

        # Create functions for optimisation
        assert budget < num_items
        self.budget = budget
        # self.opt = SubmodularOptimizer(self.get_objective, self.budget, num_iters=1)

        # Undo random seed setting
        self._set_seed()

    def _load_instances(self, num_instances, num_items, num_targets):
        return

    def _generate_features(self, Ysets):
        return

    def get_train_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs_train[self.train_idxs],
            self.Ys_train[self.train_idxs],
            torch.ones(self.num_targets).expand(len(self.train_idxs), -1),
        )

    def get_val_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs_train[self.val_idxs],
            self.Ys_train[self.val_idxs],
            torch.ones(self.num_targets).expand(len(self.val_idxs), -1),
        )

    def get_test_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs_test,
            self.Ys_test,
            torch.ones(self.num_targets).expand(len(self.Ys_test), -1),
        )

    def get_model_shape(self):
        return self.num_targets, self.num_targets

    def get_output_activation(self):
        return "relu"

    def get_twostageloss(self):
        return "mse"

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        obj = 0
        return obj

    def get_decision(self, Y, params, ptoSolver=None, Z_init=None, **kwargs):
        final_sol, final_obj = 0, 0
        return final_sol, final_obj

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "n_vars": self.Ys_train.shape[1],
            "get_objective": self.get_objective,
            "budget": self.budget
        }
