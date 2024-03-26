import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module
from scipy.spatial import distance

from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem


class TSP(PTOProblem):
    """ """

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        prob_version="gen",
        **kwargs,
    ):
        super(TSP, self).__init__()
        self._set_seed(rand_seed)
        if prob_version == "gen":
            n_nodes, n_features = kwargs["num_nodes"], kwargs["num_features"]
            self.n_nodes = n_nodes
            self.load_dataset(
                num_train_instances, num_test_instances, n_nodes, n_features, rand_seed
            )

    def get_train_data(self, **kwargs):
        return self.Xs_train, self.Ys_train, self.Ys_train

    def get_val_data(self, **kwargs):
        return self.Xs_val, self.Ys_val, self.Ys_val

    def get_test_data(self, **kwargs):
        return self.Xs_test, self.Ys_test, self.Ys_test

    def get_model_shape(self):
        return self.Xs_train.shape[-1], self.Ys_train.shape[-1]

    def get_output_activation(self):
        return "identity"

    def get_twostageloss(self):
        return "mse"

    def get_decision(self, Y, params, optSolver, isTrain=True, **kwargs):
        if torch.is_tensor(Y):
            Y = Y.cpu()
        else:
            Y = to_tensor(Y)

        sol, obj = [], []
        print("Y", Y)
        for i in range(len(Y)):
            # solve
            solp, objp, other = optSolver.solve(Y[i])
            sol.append(solp)
            obj.append(objp)
        sols_array, objs_array = np.array(sol), np.array(obj)
        return sols_array, objs_array

    def get_objective(self, Y, Z, **kwargs):
        return Y * Z

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "n_nodes": self.n_nodes,
        }

    def load_dataset(
        self, num_train_instances, num_test_instances, num_nodes, num_features, rand_seed
    ):
        train_feats, train_costs = self.genData(
            num_train_instances,
            num_features,
            num_nodes,
            deg=1,
            noise_width=0,
            seed=rand_seed,
        )
        val_feats, val_costs = self.genData(
            num_train_instances,
            num_features,
            num_nodes,
            deg=1,
            noise_width=0,
            seed=rand_seed,
        )
        test_feats, test_costs = self.genData(
            num_test_instances,
            num_features,
            num_nodes,
            deg=1,
            noise_width=0,
            seed=rand_seed,
        )
        self.Xs_train, self.Ys_train = train_feats, train_costs
        self.Xs_val, self.Ys_val = val_feats, val_costs
        self.Xs_test, self.Ys_test = test_feats, test_costs
        return

    @staticmethod
    def genData(num_data, num_features, num_nodes, deg=1, noise_width=0, seed=135):
        """
        A function to generate synthetic data and features for travelling salesman

        Args:
            num_data (int): number of data points
            num_features (int): dimension of features
            num_nodes (int): number of nodes
            deg (int): data polynomial degree
            noise_width (float): half witdth of data random noise
            seed (int): random seed

        Returns:
            tuple: data features (np.ndarray), costs (np.ndarray)
        """
        # positive integer parameter
        if type(deg) is not int:
            raise ValueError("deg = {} should be int.".format(deg))
        if deg <= 0:
            raise ValueError("deg = {} should be positive.".format(deg))
        # set seed
        rnd = np.random.RandomState(seed)
        # number of data points
        n = num_data
        # dimension of features
        p = num_features
        # number of nodes
        m = num_nodes
        # random coordinates
        coords = np.concatenate(
            (rnd.uniform(-2, 2, (m // 2, 2)), rnd.normal(0, 1, (m - m // 2, 2)))
        )
        # distance matrix
        org_dist = distance.cdist(coords, coords, "euclidean")
        # random matrix parameter B
        B = rnd.binomial(1, 0.5, (m * (m - 1) // 2, p)) * rnd.uniform(
            -2, 2, (m * (m - 1) // 2, p)
        )
        # feature vectors
        x = rnd.normal(0, 1, (n, p))
        # init cost
        c = np.zeros((n, m * (m - 1) // 2))
        for i in range(n):
            # reshape
            index = 0
            for j in range(m):
                for k in range(j + 1, m):
                    c[i, index] = org_dist[j, k]
                    index += 1
            # noise
            noise = rnd.uniform(1 - noise_width, 1 + noise_width, m * (m - 1) // 2)
            # from feature to edge
            c[i] += (
                ((np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg)
                / 3 ** (deg - 1)
            ).reshape(-1) * noise
        # rounding
        c = np.around(c, decimals=4)
        return x, c

    def genEnv(
        self,
        env_id,
        num_train_instances,
    ):
        return env_id, num_train_instances
