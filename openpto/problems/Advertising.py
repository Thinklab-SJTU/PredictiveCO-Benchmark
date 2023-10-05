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
            # load data
            self.pretrain_X, self.pretrain_Y, self.pretrain_aux = self.load_data(
                f"{data_dir}/train.pickle"
            )
            self.train_X, self.train_Y, self.train_aux = self.load_data(
                f"{data_dir}/train_mock.pickle"
            )
            self.test_X, self.test_Y, self.test_aux = self.load_data(
                f"{data_dir}/test_mock.pickle"
            )
        else:
            assert False, "Not implemented"

    def load_data(self, path, isMock=False):
        with open(path, "rb") as f:
            data = pickle.load(f)
        data["uid"].astype("int")
        labels = move_to_tensor(data["label"])
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
        out_features = move_to_tensor(np.hstack((feat1, feat2, push1, push2)))
        if isMock:
            mockchannel = data["mockchannel"].astype("int")
            realchannel = data["realchannel"].astype("int")
            m_r_channeles = move_to_tensor(np.hstack((mockchannel, realchannel)))
            return (out_features, labels, m_r_channeles)
        else:
            return out_features, labels, torch.zeros_like(labels)

    def get_pretrain_data(self, **kwargs):
        return self.pretrain_X, self.pretrain_Y

    def get_train_data(self, **kwargs):
        return self.train_X, self.train_Y, self.train_aux

    def get_val_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_test_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_model_shape(self):
        return self.train_X.shape[1], 1

    def get_output_activation(self):
        return "sigmoid"

    def get_twostageloss(self):
        return "bce"

    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
        sols = optSolver.solve(Y, params)
        objs = self.get_objective(Y, sols)
        return sols, objs

    def get_objective(self, Y, Z, **kwargs):
        return (Y * Z).sum(-1)

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "avg_budget": self.avg_budget,
        }
