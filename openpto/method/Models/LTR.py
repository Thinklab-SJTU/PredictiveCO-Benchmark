#!/usr/bin/env python
# coding: utf-8
"""
Learning to rank Losses
"""

import numpy as np
import torch
import torch.nn.functional as F

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_tensor


class pointwiseLTR(optModel):

    def __init__(self, ptoSolver, **kwargs):
        return
    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        loss = 0
        return loss


class pairwiseLTR(optModel):

    def __init__(self, ptoSolver, **kwargs):
        super().__init__(ptoSolver)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        loss = 0
        return loss


class listwiseLTR(optModel):

    def __init__(self, ptoSolver, tau=1.0, **kwargs):
        super().__init__(ptoSolver)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        loss = 0
        return loss
