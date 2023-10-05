import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.problems.PTOProblem import PTOProblem


class Knapsack(PTOProblem):
    """
    Knapsack problem
    """

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_items=100,  # number of targets to consider
        val_frac=0.2,  # fraction of training data reserved for validation
        generate_weight=True,
        unit_weight=False,
        kfold=[0],
        noise_level=0,
        rand_seed=0,  # for reproducibility
        prob_version="gen",  # "energy" or "gen"
        knapsack_dim=1,
        num_features=5,
        poly_deg=1,
        noise_width=0,
        capacity=1,
        data_dir="./openpto/data/",
        kwargs={},
    ):
        super(Knapsack, self).__init__(data_dir)
        self.capacity = capacity
        self.num_items = num_items
        self.prob_version = prob_version
        self._set_seed(rand_seed)
        # Obtain data
        if prob_version == "energy":
            raise NotImplementedError
        elif prob_version == "gen":
            weights, feats, profits = self.genData(
                num_train_instances + num_test_instances,
                num_features,
                num_items,
                dim=knapsack_dim,
                poly_deg=poly_deg,
                noise_width=noise_width,
                seed=rand_seed,
            )
            train_feats, test_feats = (
                feats[:num_train_instances],
                feats[num_train_instances:],
            )
            train_profits, test_profits = (
                profits[:num_train_instances],
                profits[num_train_instances:],
            )
            # train set
            self.weights = weights
            self.params_train = weights.unsqueeze(0).expand(num_train_instances, -1)
            self.Xs_train, self.Ys_train = (
                train_feats,
                train_profits,
            )  # (bz, feature_dim), (bz, n_items)
            # test set
            self.params_test = weights.unsqueeze(0).expand(num_test_instances, -1)
            self.Xs_test, self.Ys_test = test_feats, test_profits
            # Split training data into train/val
            assert 0 < val_frac < 1
            self.val_idxs = range(0, int(val_frac * num_train_instances))
            self.train_idxs = range(
                int(val_frac * num_train_instances), num_train_instances
            )
        else:
            raise ValueError("Not a valid problem version: {}".format(prob_version))
        # default sovler

    def get_train_data(self):
        return (
            self.Xs_train[self.train_idxs],
            self.Ys_train[self.train_idxs],
            self.params_train[self.train_idxs],
        )

    def get_val_data(self):
        return (
            self.Xs_train[self.val_idxs],
            self.Ys_train[self.val_idxs],
            self.params_train[self.val_idxs],
        )

    def get_test_data(self):
        return self.Xs_test, self.Ys_test, self.params_test

    def get_objective(self, Y, Z, **kwargs):
        # asssert shape
        assert Y.ndim == 2
        assert Z.ndim == 2
        assert Y.shape[0] == Z.shape[0]
        assert Y.shape[1] == Z.shape[1]
        # convert to device
        if torch.is_tensor(Y):
            Y = Y.cpu()
            Z = Z.cpu()
        return (Y * Z).sum(1)

    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
        if torch.is_tensor(Y):
            Y = Y.cpu()

        # determine solver
        if optSolver is None:
            if params.ndim > 1:
                params[0]
            else:
                pass
            optSolver = KPGrbSolver(**kwargs)

        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        sol, obj = [], []
        for i in range(len(Y)):
            # solve
            optSolver.setObj(Y[i])
            solp, objp, other = optSolver.solve()
            sol.append(solp)
            obj.append(objp)
        sols_array, objs_array = np.array(sol), np.array(obj)
        return sols_array, objs_array

    def init_API(self):
        return {
            "weights": self.weights,
            "capacity": self.capacity,
            "modelSense": GRB.MAXIMIZE,
        }

    def get_model_shape(self):
        if self.prob_version == "gen":
            return self.Xs_train.shape[-1], self.num_items
        else:
            return self.Xs_train.shape[-1], 1

    def get_output_activation(self):
        return None

    @staticmethod
    def genData(
        num_instances, num_features, num_items, dim=1, poly_deg=1, noise_width=0, seed=135
    ):
        #     A function to generate synthetic data and features for knapsack

        #     Args:
        #         num_instances (int): number of data points
        #         num_features (int): dimension of features
        #         num_items (int): number of items
        #         dim (int): dimension of multi-dimensional knapsack
        #         poly_deg (int): data polynomial degree
        #         noise_width (float): half witdth of data random noise
        #         seed (int): random state seed

        #     Returns:
        #     tuple: weights of items (np.ndarray), data features (np.ndarray), costs (np.ndarray)
        # positive integer parameter
        if type(poly_deg) is not int:
            raise ValueError("poly_deg = {} should be int.".format(poly_deg))
        if poly_deg <= 0:
            raise ValueError("poly_deg = {} should be positive.".format(poly_deg))
        # set seed
        rnd = np.random.RandomState(seed)
        # number of data points
        n = num_instances
        # dimension of features
        p = num_features
        # dimension of problem
        d = dim
        # number of items
        m = num_items
        # weights of items
        weights = rnd.choice(range(300, 800), size=(d, m)) / 100
        # random matrix parameter B
        B = rnd.binomial(1, 0.5, (m, p))
        # feature vectors
        feats = rnd.normal(0, 1, (n, p))
        # value of items
        profits = np.zeros((n, m), dtype=int)
        for i in range(n):
            # cost without noise
            values = (
                np.dot(B, feats[i].reshape(p, 1)).T / np.sqrt(p) + 3
            ) ** poly_deg + 1
            # rescale
            values *= 5
            values /= 3.5**poly_deg
            # noise
            epislon = rnd.uniform(1 - noise_width, 1 + noise_width, m)
            values *= epislon
            # convert into int
            values = np.ceil(values)
            profits[i, :] = values
            # float
            profits = profits.astype(np.float64)
        # TODO: currently only support 1-dim knapsack
        return (
            torch.Tensor(weights).squeeze(0),
            torch.Tensor(feats),
            torch.Tensor(profits),
        )

    def get_twostageloss(self):
        return "mse"
