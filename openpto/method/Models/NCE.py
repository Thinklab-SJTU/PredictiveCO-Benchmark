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
    """
    Code from:
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver, **kwargs)
        # solution pool
        n_vars = ptoSolver.num_vars
        self.solpool = np.empty((0, n_vars))

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # coeff_hat = coeff_hat.squeeze(-1)

        # get true solution
        sol_true, _ = problem.get_decision(
            coeff_true,
            params=params,
            ptoSolver=self.ptoSolver,
            isTrain=False,
            **problem.init_API(),
        )
        sol_true = to_tensor(sol_true).to(device)

        # obtain solution cache if empty
        if len(self.solpool) == 0:
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux,
                ptoSolver=self.ptoSolver,
                isTrain=True,
                **problem.init_API(),
            )
        # solve
        sols_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, self.ptoSolver, **problem.init_API()
        )
        # add into solpool
        self.solpool = np.concatenate((self.solpool, sols_hat))
        # remove duplicate
        self.solpool = np.unique(self.solpool, axis=0)
        solpool = to_tensor(self.solpool).to(device)

        # get obj
        # print(solpool.shape, coeff_hat.shape)
        expand_shape = torch.Size([solpool.shape[0]] + list(coeff_hat.shape[1:]))
        coeff_hat_pool = coeff_hat.expand(*expand_shape)
        obj_cp = problem.get_objective(coeff_hat, sol_true, params)
        objpool_cp = problem.get_objective(coeff_hat_pool, solpool, params)
        # get loss
        if self.ptoSolver.modelSense == GRB.MINIMIZE:
            loss = obj_cp - objpool_cp
        elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
            loss = objpool_cp - obj_cp
        else:
            raise NotImplementedError

        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss


# class contrastiveMAP(optModel):
#     """
#     An autograd module for Maximum A Posterior contrastive estimation as
#     surrogate loss functions, which is a efficient self-contrastive algorithm.

#     For the MAP, the cost vector needs to be predicted from contextual data and
#     maximizes the separation of the probability of the optimal solution.

#     Thus, allows us to design an algorithm based on stochastic gradient descent.

#     Reference:
#     """

#     def __init__(self, ptoSolver=1):
#         """
#         Args:
#             ptoSolver (optModel): an  optimization model
#
#         """
#         super().__init__(ptoSolver)
#         # solution pool
#         self.solpool = np.unique(dataset.sols.copy(), axis=0)  # remove duplicate

# def forward(self, coeff_hat, sol_true, reduction="mean"):
#     """
#     Forward pass
#     """
#     # get device
#     device = coeff_hat.device
#     # convert tensor
#     cp = coeff_hat.detach().cpu().numpy()
#     # solve
#     sols_hat, _ = _solve_in_pass(cp, self.ptoSolver, self.pool)
#     # add into solpool
#     self.solpool = np.concatenate((self.solpool, sols_hat))
#     # remove duplicate
#     self.solpool = np.unique(self.solpool, axis=0)
#     solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
#     # get current obj
#     obj_cp = torch.einsum("bd,bd->b", coeff_hat, sol_true).unsqueeze(1)
#     # get obj for solpool
#     objpool_cp = torch.einsum("bd,nd->bn", coeff_hat, solpool)
#     # get loss
#     if self.ptoSolver.modelSense == GRB.MINIMIZE:
#         loss, _ = (obj_cp - objpool_cp).max(axis=1)
#     if self.ptoSolver.modelSense == GRB.MAXIMIZE:
#         loss, _ = (objpool_cp - obj_cp).max(axis=1)
#     # reduction
#     loss = do_reduction(loss, hyperparams["reduction"])
#     return loss
