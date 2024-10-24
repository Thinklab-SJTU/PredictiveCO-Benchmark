import os
import pickle
import random
import time

from copy import deepcopy

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module
from torch.multiprocessing import Pool

from openpto.method.Models.abcOptModel import optModel

NUM_CPUS = os.cpu_count()


class LODL(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        self.obj_fn = None

    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true,
        params,
        **hyperparams,
    ):
        """
        Forward pass
        """
        if self.obj_fn is None:
            self.obj_fn = self._get_learned_loss(problem, **hyperparams)
        return self.obj_fn(coeff_hat, coeff_true, **hyperparams)

    def _get_learned_loss(
        self,
        problem,
        model_type="weightedmse",
        folder="./saved_problems",
        num_samples=400,
        sampling="random",
        sampling_std=None,
        serial=True,
        **kwargs,
    ):
        surrogate_decision_quality = 0
        return surrogate_decision_quality



