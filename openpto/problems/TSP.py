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
            self.load_dataset(
                num_train_instances,
                num_test_instances,
                n_nodes,
                n_features,
                val_frac,
                rand_seed,
                **kwargs,
            )
        elif prob_version == "gen-ood":
            n_nodes, n_features = kwargs["num_nodes"], kwargs["num_features"]
            self.n_nodes = n_nodes
            poly_deg, noise_width = kwargs["poly_deg"], kwargs["noise_width"]
            raise NotImplementedError("not implemented")
            # self.load_ood_dataset(
            #     num_train_instances,
            #     num_test_instances,
            #     n_nodes,
            #     n_features,
            #     poly_deg,
            #     noise_width,
            #     val_frac,
            #     rand_seed,
            # )
        elif prob_version == "gen-global":
            n_nodes, n_features = kwargs["num_nodes"], kwargs["num_features"]
            self.n_nodes = n_nodes
            poly_deg, noise_width = kwargs["poly_deg"], kwargs["noise_width"]
            self.load_from_global_feats(
                num_train_instances,
                num_test_instances,
                n_nodes,
                n_features,
                poly_deg,
                noise_width,
                val_frac,
                rand_seed,
                kwargs,
            )
        else:
            raise NotImplementedError("not implemented")

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
        # print("decision shape:", Y.shape, Z.shape )
        return (Y.squeeze(-1) * Z).sum(-1, keepdims=True)

    def init_API(self):
        return {
            "modelSense": GRB.MINIMIZE,
            "n_nodes": self.n_nodes,
        }

    def load_dataset(
        self,
        num_train_instances,
        num_test_instances,
        n_nodes,
        n_features,
        val_frac,
        rand_seed,
        **kwargs,
    ):
        # new version of generated data
        costs, node_feats, edge_feats = self.gendata(
            num_train_instances + num_test_instances,
            n_features,
            n_nodes,
            rand_seed,
            **kwargs,
        )
        num_train_instances + num_test_instances
        # node: B, N, 2
        # edge: B, E, d
        feats = torch.cat((node_feats, edge_feats), dim=-1)
        costs = costs.unsqueeze(-1)
        print("node_feats: ", node_feats.shape, edge_feats.shape, feats.shape)
        print("costs: ", costs.shape)
        # assert 0
        # feats = torch.cat(
        #     (node_feats.reshape(n_data, -1), edge_feats.reshape(n_data, -1)), dim=-1
        # )
        train_feats, train_costs = (
            feats[:num_train_instances],
            costs[:num_train_instances],
        )
        test_feats, test_costs = (
            feats[num_train_instances:],
            costs[num_train_instances:],
        )
        n_trains = int((1 - val_frac) * num_train_instances)
        self.Xs_train, self.Ys_train = train_feats[:n_trains], train_costs[:n_trains]
        self.Xs_val, self.Ys_val = train_feats[n_trains:], train_costs[n_trains:]
        self.Xs_test, self.Ys_test = test_feats, test_costs
        # print("train: ", self.Xs_train, "costs: ", self.Ys_train)
        # print("val: ", self.Xs_val, "costs: ", self.Ys_val)
        # print("test: ", self.Xs_test, "costs: ", self.Ys_test)
        return

    def load_from_global_feats(
        self,
        num_train_instances,
        num_test_instances,
        num_nodes,
        num_features,
        deg,
        noise_width,
        val_frac,
        rand_seed,
        **kwargs,
    ):
        # uses data genearation from pyEPO
        feats, costs = self.gen_from_global_feats(
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
        n_nodes,
        n_features,
        val_frac,
        rand_seed,
        **kwargs,
    ):
        return

    @staticmethod
    def gen_from_global_feats(
        num_data, num_features, num_nodes, deg, noise_width, seed=2023
    ):
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
        # random coordinates
        coords = np.concatenate(
            (
                rnd.uniform(-2, 2, (3, num_nodes // 2, 2)),
                rnd.normal(0, 1, (3, num_nodes - num_nodes // 2, 2)),
            ),
            axis=1,
        )
        # distance matrix
        pairwise_dist = distance.cdist(coords, coords, "euclidean")
        # random matrix parameter B
        B = rnd.binomial(
            1, 0.5, (num_nodes * (num_nodes - 1) // 2, num_features)
        ) * rnd.uniform(-2, 2, (num_nodes * (num_nodes - 1) // 2, num_features))
        # feature vectors
        x = rnd.normal(0, 1, (num_data, num_features))
        # init cost
        time = np.zeros((num_data, num_nodes * (num_nodes - 1) // 2))
        for i in range(num_data):
            # reshape
            index = 0
            for j in range(num_nodes):
                for k in range(j + 1, num_nodes):
                    time[i, index] = pairwise_dist[j, k]
                    index += 1
            # noise
            noise = rnd.uniform(
                1 - noise_width, 1 + noise_width, num_nodes * (num_nodes - 1) // 2
            )
            # from feature to edge
            time[i] += (
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
        time = np.around(time, decimals=4)
        return torch.FloatTensor(x), torch.FloatTensor(time)

    @staticmethod
    def gendata(n_data, n_feats, n_nodes, seed, **kwargs):
        def poly_func(B, input):
            n_units = input.shape[-1]
            return (np.dot(input, B) / np.sqrt(n_units) + 3) ** deg / (3 ** (deg - 1))

        n_edges = int((n_nodes * (n_nodes - 1)) / 2)
        deg, noise_width = kwargs["poly_deg"], kwargs["noise_width"]
        # set seed
        rnd = np.random.RandomState(seed)
        # random coordinates
        coords = generate_nodes_coord(n_data, n_nodes, kwargs)  # n_data, n_nodes, 2
        # distance matrix
        pairwise_dist = np.array(
            [
                distance.cdist(coords[idx], coords[idx], "euclidean")
                for idx in range(n_data)
            ]
        )
        busy_degree = np.abs(rnd.normal(1, 1, (1, n_edges)))
        # random matrix parameter B
        B = rnd.binomial(1, 0.5, (n_feats)) * rnd.uniform(-2, 2, (n_feats))
        # feature vectors
        edge_feats = rnd.normal(0, 1, (n_data, n_edges, n_feats))
        time = np.zeros((n_data, n_edges))
        node_feats = list()
        for i in range(n_data):
            incremental_idx = 0
            for j in range(n_nodes):
                for k in range(j):
                    time[i, incremental_idx] = (
                        pairwise_dist[i, j, k] * busy_degree[0, incremental_idx]
                    )
                    incremental_idx += 1
            # noise
            noise = rnd.uniform(1 - noise_width, 1 + noise_width, n_edges)
            poly_term = poly_func(B, edge_feats[i]).reshape(
                -1
            )  # B: (E,)   x[i]:(E, d) # poly_term: d
            time[i] += poly_term * noise
            node_feats.append(aggr_node2edge(coords[i]))
        node_feats = np.array(node_feats)
        return (
            torch.FloatTensor(time),
            torch.FloatTensor(node_feats),
            torch.FloatTensor(edge_feats),
        )

    def genEnv(
        self,
        env_id,
        num_train_instances,
    ):
        return env_id, num_train_instances


#########################################
#          Generate Nodes_coord         #
#########################################


def aggr_node2edge(node_feats):
    aggr_edge_feats = list()
    incremental_idx = 0
    n_nodes = node_feats.shape[0]
    for j in range(n_nodes):
        for k in range(j):
            aggr_edge_feats.append(np.hstack((node_feats[j], node_feats[k])))
            incremental_idx += 1
    return aggr_edge_feats


def generate_nodes_coord(batch_size: int, n_nodes: int, kwargs):
    if kwargs["type"] == "uniform":
        return generate_uniform(
            batch_size, n_nodes, low=kwargs["low"], high=kwargs["high"]
        )
    elif kwargs["type"] == "cluster":
        return generate_cluster(
            batch_size=batch_size,
            n_nodes=n_nodes,
            num_clusters=kwargs["num_clusters"],
            cluster_std=kwargs["cluster_std"],
            center_low=kwargs["center_low"],
            center_high=kwargs["center_high"],
        )
    elif kwargs["type"] == "gaussian":
        return generate_gaussian(
            batch_size=batch_size,
            n_nodes=n_nodes,
            mean_x=kwargs["mean_x"],
            mean_y=kwargs["mean_y"],
            std=kwargs["gaussian_std"],
        )
    # elif kwargs["type"] == "cluster_fixed_centers":
    #     return generate_cluster_fixed_centers(batch_size, n_nodes)
    else:
        raise NotImplementedError


def generate_uniform(batch_size: int, n_nodes: int, low: float, high: float):
    return np.random.uniform(low, high, (batch_size, n_nodes, 2))


def generate_gaussian(
    batch_size: int,
    n_nodes: int,
    mean_x: float = 0.0,
    mean_y: float = 0.0,
    std: float = 1.0,
):
    return np.random.normal(
        loc=[mean_x, mean_y], scale=std, size=(batch_size, n_nodes, 2)
    )


def generate_cluster(
    batch_size: int,
    n_nodes: int,
    num_clusters: int,
    cluster_std: float,
    center_low: float,
    center_high: float,
):
    nodes_coords = np.zeros([batch_size, n_nodes, 2])
    for i in range(batch_size):
        cluster_centers = np.random.uniform(center_low, center_high, (num_clusters, 2))
        cluster_points = []
        for center in cluster_centers:
            points = np.random.normal(
                loc=center, scale=cluster_std, size=(n_nodes // num_clusters, 2)
            )
            cluster_points.append(points)
        nodes_coords[i] = np.concatenate(cluster_points, axis=0)  # (5,  2) (5, 2)
    return nodes_coords


def generate_cluster_fixed_centers(batch_size: int, n_nodes: int):
    assert n_nodes in [100, 500]
    nodes_coords = np.zeros([batch_size, n_nodes, 2])
    for i in range(batch_size):
        num_clusters_axis = 2 if n_nodes == 100 else 5
        num_clusters = num_clusters_axis**2
        cluster_centers_axis = np.linspace(0, 1, num_clusters_axis * 2 + 1)[1::2]
        x, y = np.meshgrid(cluster_centers_axis, cluster_centers_axis)
        cluster_centers = [[x, y] for x, y in zip(x.flatten(), y.flatten())]
        scale = (
            1 / (num_clusters_axis * 3 * 3)
            if n_nodes == 100
            else 1 / (num_clusters_axis * 3 * 3)
        )
        cluster_points = []
        for center in cluster_centers:
            points = np.random.normal(
                loc=center, scale=scale, size=(n_nodes // num_clusters, 2)
            )
            cluster_points.append(points)
        nodes_coords[i] = np.concatenate(cluster_points, axis=0)
    return nodes_coords
