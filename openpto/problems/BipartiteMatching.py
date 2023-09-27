import pickle
import random

import networkx as nx
import numpy as np
import torch

from gurobipy import GRB

from openpto.method.Solvers.cvxpy.cp_bmatching import BmatchingSolver
from openpto.problems.PTOProblem import PTOProblem

# from SubmodularOptimizer import SubmodularOptimizer


class BipartiteMatching(PTOProblem):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=20,  # number of instances to use from the dataset to train
        num_test_instances=6,  # number of instances to use from the dataset to test
        num_nodes=50,  # number of nodes in the LHS and RHS of the bipartite matching graphs
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        prob_version="bipartitematching",
        data_dir="./openpto/data/",
    ):
        super(BipartiteMatching, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        # Load train and test labels
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        print("wule", self.num_train_instances, self.num_test_instances)
        self.num_nodes = num_nodes
        self.Xs, self.Ys = self._load_instances(
            self.num_train_instances, self.num_test_instances, self.num_nodes
        )
        # self.Xs = torch.load('data/cora_features_bipartite.pt').reshape((27, 50, 50, 2866))
        # self.Ys = torch.load('data/cora_graphs_bipartite.pt').reshape((27, 50, 50))
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
        # """
        # Loads the labels (Ys) of the prediction from a file, and returns a subset of it parameterised by instances.
        # """
        g = nx.read_edgelist("openpto/data/cora.cites")
        nodes_before = [int(v) for v in g.nodes()]

        g = nx.convert_node_labels_to_integers(g, first_label=0)
        a = np.loadtxt("openpto/data/cora_cites_metis.txt.part.27")
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

        features = np.loadtxt("openpto/data/cora.content")
        features_idx = features[:, 0]
        features = features[:, 1:]
        n_nodes = 50
        Ps = np.zeros((27, n_nodes**2))
        n_features = 1433
        data = np.zeros((27, n_nodes**2, 2 * n_features))
        msubs = np.zeros((27, n_nodes**2))
        M = np.load("openpto/data/cora.msubject.npy", allow_pickle=True)
        partition = pickle.load(open("openpto/data/cora_partition.pickle", "rb"))

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
                # lhs_nodes_idx = [list(g_part[i].nodes()).index(v) for v in lhs_nodes]
                # rhs_nodes_idx = [list(g_part[i].nodes()).index(v) for v in rhs_nodes]
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

        # with open('cora_data.pickle', 'wb') as f:
        #     pickle.dump((Ps, data, msubs), f)
        # print("->Done.")
        data_tensor = torch.Tensor(np.array(data))
        Ps_tensor = torch.Tensor(np.array(Ps)).reshape(-1, 2500)
        return data_tensor, Ps_tensor

    def get_train_data(self):
        return (
            self.Xs[self.train_idxs],
            self.Ys[self.train_idxs],
            [None for _ in range(len(self.train_idxs))],
        )

    def get_val_data(self):
        return (
            self.Xs[self.val_idxs],
            self.Ys[self.val_idxs],
            [None for _ in range(len(self.val_idxs))],
        )

    def get_test_data(self):
        return (
            self.Xs[self.test_idxs],
            self.Ys[self.test_idxs],
            [None for _ in range(len(self.test_idxs))],
        )

    def get_model_shape(self):
        return self.Xs.shape[-1], 1

    def get_output_activation(self):
        return "sigmoid"

    def get_twostageloss(self):
        return "ce"

    def get_objective(self, Y, Z, **kwargs):
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        if isinstance(Y, np.ndarray) and isinstance(Z, torch.Tensor):
            Z = np.array(Z).cpu()
        if isinstance(Y, torch.Tensor) and isinstance(Z, np.ndarray):
            Z = torch.tensor(Z).cuda()
        if isinstance(Y, torch.Tensor):
            Z = Z.cuda()
            Y = Y.cuda()
        ans_list = (Y * Z).sum(axis=1)
        return ans_list

    def get_decision(
        self,
        Y,
        params,
        optSolver=None,
        isTrain=False,
        max_instances_per_batch=5000,
        **kwargs,
    ):
        # Split Y into reasonably sized chunks so that we don't run into memory issues
        # Assumption Y is only 3D at max
        Y = Y.reshape(-1, self.num_nodes, self.num_nodes)
        flag_numpy = 0
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
            flag_numpy = 1
        ins_num = len(Y)
        sols = []
        for i in range(ins_num):
            # solve
            if isTrain:
                sol = self.opt_train(Y[i])
            else:
                sol = self.opt_test(Y[i])
            sol = sol[0].cpu().numpy().reshape(-1)
            sols.append(sol)
        if flag_numpy == 1:
            sols = np.array(sols)
        else:
            sols = torch.tensor(sols)
        Y = Y.reshape(-1, self.num_nodes * self.num_nodes)
        objs = self.get_objective(Y, sols)
        return sols, objs

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
        }

    # def _create_constraint_matrix(self):
    #     """
    #     Creates a matrix representation for the inequality constraints
    #     Gx <= h specific to the bipartite matching problem.

    #     Variables:
    #     #LHS nodes = #RHS nodes = self.num_nodes
    #     n_vars: (num_nodes * num_nodes) vector
    #     n_constraints = n_vars + #LHS nodes + #RHS nodes

    #     Output:
    #     G: (n_constraints, n_vars) matrix
    #     h: (n_constraints,) vector
    #     """
    #     # Create flow constraints
    #     #   Create constraints for RHS nodes
    #     #   \sum_i z_ij <= 1, forall j
    #     G_lhs = torch.eye(self.num_nodes).repeat(1, self.num_nodes)
    #     h_lhs = torch.ones(self.num_nodes)

    #     #   Create constraints for LHS nodes
    #     #   \sum_j z_ij <= 1, forall i
    #     G_rhs = G_lhs.detach().clone()
    #     G_rhs = G_rhs.reshape((self.num_nodes, self.num_nodes, self.num_nodes)).permute((0, 2, 1)).reshape((self.num_nodes, -1))
    #     h_rhs = torch.ones(self.num_nodes)

    #     # Create non-negativity constraints
    #     G_ineq = -torch.eye(self.num_nodes * self.num_nodes)
    #     h_ineq = torch.zeros(self.num_nodes * self.num_nodes)

    #     # Putting them together
    #     G = torch.cat((G_lhs, G_rhs, G_ineq))
    #     h = torch.cat((h_lhs, h_rhs, h_ineq))

    #     return G, h


if __name__ == "__main__":
    # pdb.set_trace()
    problem = BipartiteMatching()
