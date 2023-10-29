import random
import time

from abc import ABC, abstractmethod

import numpy as np
import torch


class PTOProblem(ABC):
    """A class that defines an arbitrary predict-then-optimise problem."""

    def __init__(self, data_dir="./openpto/data/", **kwargs):
        super(PTOProblem, self).__init__()
        self.data_dir = data_dir

    @abstractmethod
    def get_train_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_val_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_model_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def get_output_activation(self):
        raise NotImplementedError()

    @abstractmethod
    def get_twostageloss(self):
        raise NotImplementedError()

    @abstractmethod
    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def init_API(self):
        return dict()

    def get_eval_metric(self):
        return "regret"

    def _set_seed(self, rand_seed=int(time.time())):
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
