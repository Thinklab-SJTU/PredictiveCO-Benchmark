#!/usr/bin/env python
# coding: utf-8
"""
"""

import torch

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Models.qpthlocal.qp import QPFunction, QPSolvers


class QPTL(optModel):
    """ """

    def __init__(self, optSolver, processes=1, solve_ratio=1, lambd=0.1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)
        # smoothing parameter
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = lambd

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
        Q = torch.eye(n_items) / tau
        # G = torch.cat((torch.from_numpy(weights).float(), torch.diagflat(torch.ones(n_items)),
        # torch.diagflat(torch.ones(n_items)*-1)), 0)
        # h = torch.cat((torch.tensor([capacity],dtype=torch.float),torch.ones(n_items),torch.zeros(n_items)))

        G = torch.from_numpy(weights).float()
        h = torch.tensor([capacity], dtype=torch.float)

        c_true = - coeff_true
        c_pred = - coeff_hat
        solver = QPFunction(
            verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad
        )
        x = solver(
            Q.expand(n_train, *Q.shape),
            c_pred.squeeze(),
            G.expand(n_train, *G.shape),
            h.expand(n_train, *h.shape),
            torch.Tensor(),
            torch.Tensor(),
        )
        loss = (x.squeeze() * c_true).mean()
        return loss
