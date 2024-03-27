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
        num_train_instances=400,  # number of instances to use from the dataset to train
        num_test_instances=200,  # number of instances to use from the dataset to test
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
            poly_deg, noise_width = kwargs["poly_deg"], kwargs["noise_width"]
            self.load_dataset(
                num_train_instances,
                num_test_instances,
                n_nodes,
                n_features,
                poly_deg,
                noise_width,
                val_frac,
                rand_seed,
            )
        elif prob_version == "gen-ood":
            n_nodes, n_features = kwargs["num_nodes"], kwargs["num_features"]
            self.n_nodes = n_nodes
            poly_deg, noise_width = kwargs["poly_deg"], kwargs["noise_width"]
            self.load_dataset(
                num_train_instances,
                num_test_instances,
                n_nodes,
                n_features,
                poly_deg,
                noise_width,
                val_frac,
                rand_seed,
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

    def get_decision(self, Y, params, ptoSolver, isTrain=True, **kwargs):
        if torch.is_tensor(Y):
            Y = Y.cpu()
        else:
            Y = to_tensor(Y)

        sols = []
        for i in range(len(Y)):
            # solve
            solp, _, _ = ptoSolver.solve(Y[i])
            sols.append(solp)
        sols = torch.FloatTensor(sols)
        objs = self.get_objective(Y, sols, params, **kwargs)
        return sols, objs

    def get_objective(self, Y, Z, aux_data, **kwargs):
        return (Y * Z).sum(-1, keepdims=True)

    def init_API(self):
        return {
            "modelSense": GRB.MINIMIZE,
            "n_nodes": self.n_nodes,
        }

    def load_dataset(
        self,
        num_train_instances,
        num_test_instances,
        num_nodes,
        num_features,
        deg,
        noise_width,
        val_frac,
        rand_seed,
    ):
        feats, costs = self.genData(
            num_train_instances + num_test_instances,
            num_features,
            num_nodes,
            deg=deg,
            noise_width=noise_width,
            seed=rand_seed,
        )
        train_feats, train_costs = (
            feats[:num_train_instances],
            costs[:num_train_instances],
        )
        test_feats, test_costs = feats[num_train_instances:], costs[num_train_instances:]
        n_trains = int((1 - val_frac) * num_train_instances)
        self.Xs_train, self.Ys_train = train_feats[:n_trains], train_costs[:n_trains]
        self.Xs_val, self.Ys_val = train_feats[n_trains:], train_costs[n_trains:]
        self.Xs_test, self.Ys_test = test_feats, test_costs
        print("train: ", self.Xs_train, "costs: ", self.Ys_train)
        print("val: ", self.Xs_val, "costs: ", self.Ys_val)
        print("test: ", self.Xs_test, "costs: ", self.Ys_test)
        return

    def load_ood_dataset(
        self,
        num_train_instances,
        num_test_instances,
        num_nodes,
        num_features,
        val_frac,
        rand_seed,
        **kwargs,
    ):
        return

    @staticmethod
    def genData(num_data, num_features, num_nodes, deg, noise_width, seed=2023):
        """
        Adopted by PyEPO
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
        # random coordinates
        coords = np.concatenate(
            (
                rnd.uniform(-2, 2, (num_nodes // 2, 2)),
                rnd.normal(0, 1, (num_nodes - num_nodes // 2, 2)),
            )
        )
        # distance matrix
        org_dist = distance.cdist(coords, coords, "euclidean")
        # random matrix parameter B
        B = rnd.binomial(
            1, 0.5, (num_nodes * (num_nodes - 1) // 2, num_features)
        ) * rnd.uniform(-2, 2, (num_nodes * (num_nodes - 1) // 2, num_features))
        # feature vectors
        x = rnd.normal(0, 1, (num_data, num_features))
        # init cost
        c = np.zeros((num_data, num_nodes * (num_nodes - 1) // 2))
        for i in range(num_data):
            # reshape
            index = 0
            for j in range(num_nodes):
                for k in range(j + 1, num_nodes):
                    c[i, index] = org_dist[j, k]
                    index += 1
            # noise
            noise = rnd.uniform(
                1 - noise_width, 1 + noise_width, num_nodes * (num_nodes - 1) // 2
            )
            # from feature to edge
            c[i] += (
                (
                    (
                        np.dot(B, x[i].reshape(num_features, 1)).T / np.sqrt(num_features)
                        + 3
                    )
                    ** deg
                )
                / 3 ** (deg - 1)
            ).reshape(-1) * noise
        # rounding
        c = np.around(c, decimals=4)
        return torch.FloatTensor(x), torch.FloatTensor(c)

    def genEnv(
        self,
        env_id,
        num_train_instances,
    ):
        return env_id, num_train_instances
