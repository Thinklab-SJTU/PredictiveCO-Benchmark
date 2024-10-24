#!/usr/bin/env python
# coding: utf-8
"""
Noise contrastive estimation loss function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_tensor


class NCE(optModel):
    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver, **kwargs)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        loss = 0 
        return loss
