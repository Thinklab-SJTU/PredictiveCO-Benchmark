#!/usr/bin/env python
# coding: utf-8
"""
cvxpy layer
"""

import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel


class cpLayer(optModel):
    """
    Reference:
    """

    def __init__(self, optSolver, processes=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores

        """
        super().__init__(optSolver, processes)

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
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(objs_hat)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(objs_hat)
        elif hyperparams["reduction"] == "none":
            loss = objs_hat
        else:
            raise ValueError(f"No reduction {hyperparams['reduction']}.")

        if self.optSolver.modelSense == GRB.MINIMIZE:
            pass
        elif self.optSolver.modelSense == GRB.MAXIMIZE:
            loss = -loss

        return loss
