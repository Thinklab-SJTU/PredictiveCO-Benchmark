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
        """
        Loads the labels (Ys) of the prediction from a file, and returns a subset of it parameterised by instances.
        """
        # Load the dataset
        with open(f"{self.data_dir}/budget_allocation_data.pkl", "rb") as f:
            Yfull, _ = pickle.load(f, encoding="bytes")
        Yfull = np.array(Yfull)  # [1000,100, 500]

        # Whittle the dataset down to the right size
        def whittle(matrix, size, dim):
            assert size <= matrix.shape[dim]
            elements = np.random.choice(matrix.shape[dim], size)
            return np.take(matrix, elements, axis=dim)

        Ys = whittle(Yfull, num_instances, 0)
        Ys = whittle(Ys, num_items, 1)
        Ys = whittle(Ys, num_targets, 2)

        return torch.from_numpy(Ys).float().detach()

    def _generate_features(self, Ysets):
        """
        Converts labels (Ys) + random noise, to features (Xs)
        """
        # Generate random matrix common to all Ysets (train + test)
        transform_nn = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, self.num_targets)
        )

        # Generate training data by scrambling the Ys based on this matrix
        Xsets = []
        for Ys in Ysets:
            # Normalise data across the last dimension
            Ys_mean = Ys.reshape((-1, Ys.shape[2])).mean(dim=0)
            Ys_std = Ys.reshape((-1, Ys.shape[2])).std(dim=0)
            Ys_standardised = (Ys - Ys_mean) / (Ys_std + 1e-10)
            assert not torch.isnan(Ys_standardised).any()

            # Add noise to the data to complicate prediction
            fake_features = torch.normal(
                mean=torch.zeros(Ys.shape[0], Ys.shape[1], self.num_fake_targets)
            )
            Ys_augmented = torch.cat((Ys_standardised, fake_features), dim=2)
            print("Ys_augmented: ", Ys_augmented.shape)
            print("transform_nn: ", transform_nn)
            # Encode Ys as features by multiplying them with a random matrix
            Xs = transform_nn(Ys_augmented).detach().clone()
            Xsets.append(Xs)

        return (*Xsets,)

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
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        Y = to_tensor(Y).cpu()
        Z = to_tensor(Z).cpu()
        # TODO: check maximize or minimize
        assert Y.shape[-2] == Z.shape[-1]
        assert Z.ndim + 1 == Y.ndim

        # Initialise weights to default value
        w = None  # TODO: support customizing weights
        if w is None:
            w = torch.ones(Y.shape[-1])
        else:
            assert Y.shape[-1] == w.shape[0]
            assert len(w.shape) == 1

        # Calculate objective
        p_fail = 1 - Z.unsqueeze(-1) * Y
        p_all_fail = p_fail.prod(dim=-2)
        obj = (w * (1 - p_all_fail)).sum(dim=-1)
        return obj

    def get_decision(self, Y, params, ptoSolver=None, Z_init=None, **kwargs):
        assert Y.ndim == 3
        if Z_init is None:
            Z_init = torch.rand(Y.shape[1:-1])
        Z_init = to_tensor(Z_init).to(self.device)
        Y = to_tensor(Y).to(self.device)
        Z = torch.cat([ptoSolver.solve(y, Z_init=Z_init) for y in Y], dim=0).view(
            (*Y.shape[:-2], -1)
        )
        # Z = torch.ones(*Y.shape[:-1])
        final_sol = Z.cpu()
        final_obj = self.get_objective(Y, Z)
        return final_sol, final_obj

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "n_vars": self.Ys_train.shape[1],
            "get_objective": self.get_objective,
            "budget": self.budget
            # "sol_shape":torch.ones(self.Ys_train.shape[1]).shape,
        }


# Unit test for RandomTopK
if __name__ == "__main__":
    filename = "./saved_problems/budgetalloc_0.pkl"
    with open(filename, "rb") as file:
        problem = pickle.load(file)
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()
