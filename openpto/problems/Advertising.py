import pickle

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.utils_method import move_to_tensor
from openpto.problems.PTOProblem import PTOProblem


class Advertising(PTOProblem):
    """ """

    def __init__(self, avg_budget, data_dir, prob_version="real", **kwargs):
        super(Advertising, self).__init__(data_dir)
        self.avg_budget = avg_budget
        if prob_version == "real":
            self.cost_pv = [0, 0.5, 1, 1.5]
            # load data
            self.pretrain_X, self.pretrain_Y, self.pretrain_aux = self.load_data(
                f"{data_dir}/train.pickle"
            )
            self.train_X, self.train_Y, self.train_aux = self.load_data(
                f"{data_dir}/train_mock.pickle", isMock=True
            )
            self.test_X, self.test_Y, self.test_aux = self.load_data(
                f"{data_dir}/test_mock.pickle", isMock=True
            )
        else:
            assert False, "Not implemented"

    def load_data(self, path, isMock=False):
        with open(path, "rb") as f:
            data = pickle.load(f)
        data["uid"].astype("int")
        labels = move_to_tensor(data["label"]).unsqueeze(0).unsqueeze(-1)
        push_histories = data["push_history"].astype("float32")
        # data["channels_his"]
        # TODO: add channel his to features
        features = data["feat_his"].astype("float32")
        # process
        feat1 = features[:, :41]
        feat2 = features[:, 41:]
        push1 = push_histories[:, 0]
        push2 = push_histories[:, 1]
        # final
        out_features = move_to_tensor(np.hstack((feat1, feat2, push1, push2))).unsqueeze(
            0
        )
        if isMock:
            mockchannel = move_to_tensor(data["mockchannel"])
            realchannel = move_to_tensor(data["realchannel"])
            m_r_channeles = torch.vstack((mockchannel, realchannel)).long().unsqueeze(0)
            return out_features, labels, m_r_channeles
        else:
            return out_features, labels, torch.zeros_like(labels)

    def get_pretrain_data(self, **kwargs):
        return self.pretrain_X, self.pretrain_Y, self.pretrain_aux

    def get_train_data(self, **kwargs):
        return self.train_X, self.train_Y, self.train_aux

    def get_val_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_test_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_model_shape(self):
        return self.train_X.shape[-1], 1

    def get_output_activation(self):
        return "sigmoid"

    def get_twostageloss(self):
        return "bce"

    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
        if torch.is_tensor(Y):
            Y = Y.cpu()
        total_budget = self.avg_budget * Y.shape[1]
        sols = optSolver.solve(Y, self.cost_pv, total_budget)
        objs = self.get_objective(Y, sols)
        return sols, objs

    def get_objective(self, Y, Z, **kwargs):
        if torch.is_tensor(Y):
            Y, Z = Y.cpu(), Z.cpu()
            return torch.sum(torch.mul(Y.reshape(1, -1, 4), Z))
        elif isinstance(Y, np.ndarray):
            return np.sum(np.multiply(Y.reshape(1, -1, 4), Z))

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "avg_budget": self.avg_budget,
        }

    def get_eval_metric(self):
        return "treatment"
