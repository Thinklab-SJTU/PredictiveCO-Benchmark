import random
import time

import numpy as np
import torch

from openpto.problems.PTOProblem import PTOProblem


class DiverseRec(PTOProblem):
    """ """

    def __init__(self, data_dir="./openpto/data/", **kwargs):
        super(DiverseRec, self).__init__()
        self.data_dir = data_dir

    def get_train_data(self, **kwargs):
        raise NotImplementedError()

    def get_val_data(self, **kwargs):
        raise NotImplementedError()

    def get_test_data(self, **kwargs):
        raise NotImplementedError()

    def get_model_shape(self):
        raise NotImplementedError()

    def get_output_activation(self):
        raise NotImplementedError()

    def get_twostageloss(self):
        raise NotImplementedError()

    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        raise NotImplementedError()

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        raise NotImplementedError()

    def init_API(self):
        return dict()

    def _set_seed(self, rand_seed=int(time.time())):
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
