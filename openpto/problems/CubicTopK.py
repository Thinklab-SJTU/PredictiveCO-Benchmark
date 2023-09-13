import pdb
import random

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.neural.RMABSolver import TopK_custom
from openpto.problems.PTOProblem import PTOProblem


class CubicTopK(PTOProblem):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=100,  # number of instances to use from the dataset to test
        num_items=50,  # number of targets to consider
        budget=2,  # number of items that can be picked
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        prob_version="gen",
        data_dir="./openpto/data/",
    ):
        super(CubicTopK, self).__init__(data_dir)
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        train_seed, test_seed = random.randrange(2**32), random.randrange(2**32)
        if prob_version == "gen":
            # Generate Dataset
            #   Save relevant parameters
            self.num_items = num_items
            self.num_train_instances = num_train_instances
            self.num_test_instances = num_test_instances
            #   Generate features
            self._set_seed(train_seed)
            self.Xs_train = (
                2 * torch.rand(self.num_train_instances, self.num_items, 1) - 1
            )
            self._set_seed(test_seed)
            self.Xs_test = 2 * torch.rand(self.num_test_instances, self.num_items, 1) - 1
            #   Generate Labels
            self.Ys_train = 10 * (self.Xs_train.pow(3) - 0.65 * self.Xs_train)
            self.Ys_test = 10 * (self.Xs_test.pow(3) - 0.65 * self.Xs_test)
        else:
            raise NotImplementedError
        # Split training data into train/val
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(
            int(self.val_frac * num_train_instances), num_train_instances
        )
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Save variables for optimisation
        assert budget < num_items
        self.budget = budget

        # Undo random seed setting
        self._set_seed()

    def get_train_data(self):
        return (
            self.Xs_train[self.train_idxs],
            self.Ys_train[self.train_idxs],
            [None for _ in range(len(self.train_idxs))],
        )

    def get_val_data(self):
        return (
            self.Xs_train[self.val_idxs],
            self.Ys_train[self.val_idxs],
            [None for _ in range(len(self.val_idxs))],
        )

    def get_test_data(self):
        return self.Xs_test, self.Ys_test, [None for _ in range(len(self.Ys_test))]

    def opt_train(self, Y):
        gamma = TopK_custom(self.budget)(-Y).squeeze()
        Z = gamma[..., 0] * Y.shape[-1]
        return Z

    def opt_test(self, Y):
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
        _, idxs = torch.topk(Y, self.budget)
        Z = torch.nn.functional.one_hot(idxs, Y.shape[-1]).sum(dim=-2).sum(-1)
        # return Z if self.budget == 0 else Z.sum(dim=-2)
        output_sols = Z.cpu().numpy()
        output_vals = self.get_objective(Y, Z)
        return output_sols, output_vals

    def get_objective(self, Y, Z, **kwargs):
        if isinstance(Z, np.ndarray):
            Z = np.expand_dims(Z, -1)
        else:
            Z = Z.unsqueeze(-1)
        return (Z * Y).sum(-1)

    def get_decision(self, Y, params, isTrain=False, **kwargs):
        return self.opt_test(Y)

    def get_model_shape(self):
        return 1, 1

    def get_output_activation(self):
        return None

    def get_twostageloss(self):
        return "mse"

    def init_API(self):
        return {"modelSense": GRB.MINIMIZE, "n_vars": self.Ys_train.shape[1]}


# Unit test for RandomTopK
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load An Example Instance
    pdb.set_trace()
    problem = CubicTopK()

    # Plot It
    Xs = problem.Xs_train.flatten().tolist()
    Ys = problem.Ys_train.flatten().tolist()
    plt.scatter(Xs, Ys)
    plt.show()
