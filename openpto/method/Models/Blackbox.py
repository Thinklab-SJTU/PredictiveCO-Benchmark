#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Black-box optimization function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import (
    do_reduction,
    minus,
    to_array,
    to_device,
    to_tensor,
)


class blackboxSolver(optModel):
    """ """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)

    def forward(
        self,
        problem,
        coeff_hat,
        params,
        **hyperparams,
    ):
        loss = 0
        return loss


class subopt_blackbox(optModel):
    """ """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)

    def forward(
        self,
        problem,
        coeff_hat,
        params,
        **hyperparams,
    ):
        loss = 0
        return loss

