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

    def __init__(self, optSolver, **kwargs):
        """ """
        super().__init__(optSolver)

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
        sols_hat, _ = problem.get_decision(
            coeff_hat,
            params=params,
            optSolver=self.optSolver,
            isTrain=True,
            **problem.init_API(),
        )
        objs_hat = problem.get_objective(coeff_hat, sols_hat, params, **hyperparams)

        # reduction
        loss = do_reduction(objs_hat, hyperparams["reduction"])

        if self.optSolver.modelSense == GRB.MINIMIZE:
            pass
        elif self.optSolver.modelSense == GRB.MAXIMIZE:
            loss = -loss

        return loss
