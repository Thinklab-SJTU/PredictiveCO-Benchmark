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
        # self.Xs = torch.load('data/cora_features_bipartite.pt').reshape((27, 50, 50, 2866))
        # self.Ys = torch.load('data/cora_graphs_bipartite.pt').reshape((27, 50, 50))
        # Split data into train/val/test
        print("training y does have negative:", sum(sum(self.Ys < 0)))
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
        # """
        # Loads the labels (Ys) of the prediction from a file, and returns a subset of it parameterised by instances.
        # """
        g = nx.read_edgelist(f"{self.data_dir}/cora.cites")
        nodes_before = [int(v) for v in g.nodes()]

        g = nx.convert_node_labels_to_integers(g, first_label=0)
        a = np.loadtxt(f"{self.data_dir}/cora_cites_metis.txt.part.27")
        g_part = []
        for i in range(27):
            g_part.append(nx.Graph(nx.subgraph(g, list(np.where(a == i))[0])))

        nodes_available = []
        for i in range(27):
            if len(g_part[i]) > 100:
                degrees = [g_part[i].degree(v) for v in g_part[i]]
                order = np.argsort(degrees)
                nodes = np.array(g_part[i].nodes())
                num_remove = len(g_part[i]) - 100
                to_remove = nodes[order[:num_remove]]
                g_part[i].remove_nodes_from(to_remove)
                nodes_available.extend(to_remove)

        for i in range(27):
            if len(g_part[i]) < 100:
                num_needed = 100 - len(g_part[i])
                to_add = nodes_available[:num_needed]
                nodes_available = nodes_available[num_needed:]
                g_part[i].add_nodes_from(to_add)

        for i in range(27):
            g_part.append(nx.subgraph(g, list(np.where(a == i))[0]))

        features = np.loadtxt(f"{self.data_dir}/cora.content")
        features_idx = features[:, 0]
        features = features[:, 1:]
        n_nodes = 50
        Ps = np.zeros((27, n_nodes**2))
        n_features = 1433
        data = np.zeros((27, n_nodes**2, 2 * n_features))
        msubs = np.zeros((27, n_nodes**2))
        M = np.load(f"{self.data_dir}/cora.msubject.npy", allow_pickle=True)
        partition = pickle.load(open(f"{self.data_dir}/cora_partition.pickle", "rb"))

        percent_removed = []
        for i in range(27):
            lhs_nodes, rhs_nodes = partition[i]
            lhs_nodes_idx = []
            rhs_nodes_idx = []
            gnodes = list(g_part[i].nodes())

            to_add = set([i for i in range(len(gnodes))])
            for v in lhs_nodes:
                try:
                    lhs_nodes_idx.append(gnodes.index(v))
                    to_add.remove(gnodes.index(v))
                except Exception:
                    print(v, " not in lhs list")
            for v in rhs_nodes:
                try:
                    rhs_nodes_idx.append(gnodes.index(v))
                    to_add.remove(gnodes.index(v))
                except Exception:
                    print(v, " not in list")
            missing_list = (
                lhs_nodes_idx
                if len(lhs_nodes_idx) < len(rhs_nodes_idx)
                else rhs_nodes_idx
            )

            while len(missing_list) < 50:
                misidx = to_add.pop()
                print("node {} idx added successfully".format(gnodes[misidx]))
                missing_list.append(misidx)
            assert len(lhs_nodes_idx) == len(rhs_nodes_idx)
            adj = nx.to_numpy_array(g_part[i])
            sum_before = adj.sum()
            adj = adj[lhs_nodes_idx]
            adj = adj[:, rhs_nodes_idx]
            edges_before = sum_before / 2
            # print(sum_before/2, adj.sum())
            percent_removed.append((edges_before - adj.sum()) / edges_before)
            Ps[i] = adj.flatten()
            # print(Ps[i].sum)
            msubs[i] = M[lhs_nodes_idx][:, rhs_nodes_idx].flatten()
            node_ids_lhs = [nodes_before[v] for v in lhs_nodes]
            node_ids_rhs = [nodes_before[v] for v in rhs_nodes]
            curr_data_idx = 0
            for j, nid in enumerate(node_ids_lhs):
                row_idx_j = int(np.where(features_idx == nid)[0][0])
                for k, nid_other in enumerate(node_ids_rhs):
                    row_idx_k = int(np.where(features_idx == nid_other)[0][0])
                    data[i, curr_data_idx, :n_features] = features[row_idx_j]
                    data[i, curr_data_idx, n_features:] = features[row_idx_k]
                    curr_data_idx += 1

        data_tensor = torch.from_numpy(data.astype(np.float32))
        Ps_tensor = torch.from_numpy(Ps.astype(np.float32)).reshape(-1, 2500, 1)
        return data_tensor, Ps_tensor

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
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        assert Y.ndim == 3
        assert Z.ndim == 2
        # convert Z to Y type
        if isinstance(Y, np.ndarray) and isinstance(Z, torch.Tensor):
            Z = to_array(Z)
        if isinstance(Y, torch.Tensor) and isinstance(Z, np.ndarray):
            Z = to_tensor(Z)
        #
        if torch.is_tensor(Y):
            Z = Z.to(self.device)
            Y = Y.to(self.device)
        ans_list = (Y.squeeze(-1) * Z).sum(axis=1)
        return ans_list

    def get_decision(
        self,
        Y,
        params,
        ptoSolver=None,
        isTrain=False,
        max_instances_per_batch=5000,
        **kwargs,
    ):
        # Split Y into reasonably sized chunks so that we don't run into memory issues
        # Assumption Y is only 3D at max
        if torch.is_tensor(Y):
            Y = Y.cpu()
        Y_unflatten = Y.reshape(-1, self.num_nodes, self.num_nodes)
        flag_numpy = 0
        if isinstance(Y_unflatten, np.ndarray):
            Y_unflatten = torch.from_numpy(Y_unflatten)
            flag_numpy = 1
        sols = []
        for i in range(len(Y_unflatten)):
            # solve
            if isTrain:
                sol = self.opt_train(Y_unflatten[i])
            else:
                sol = self.opt_test(Y_unflatten[i])
            sol = sol[0].cpu().reshape(-1)
            sols.append(sol)
        sols = torch.vstack(sols)
        if flag_numpy:
            sols = sols.numpy()
        # Y = Y.reshape(-1, self.num_nodes * self.num_nodes)
        objs = self.get_objective(Y, sols, params)
        return sols, objs

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
        }


if __name__ == "__main__":
    problem = BipartiteMatching()
