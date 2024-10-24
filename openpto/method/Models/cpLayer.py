#!/usr/bin/env python
# coding: utf-8
"""
cvxpy layer
"""


from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction


class cpLayer(optModel):
    """
    Reference:
    """

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
        """
        Forward pass
        """
        loss = 0 
        return loss
