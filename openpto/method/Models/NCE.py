#!/usr/bin/env python
# coding: utf-8
"""
Noise contrastive estimation loss function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Solvers.utils_solver import _solve_in_pass


class NCE(optModel):
    """
    Code from:
    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio, **kwargs)
        # solution pool
        n_vars = optSolver.num_vars
        self.solpool = np.empty((0, n_vars))

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # get true solution
        sol_true, _ = problem.get_decision(
            coeff_true,
            params=params,
            optSolver=self.optSolver,
            isTrain=False,
            **problem.init_API(),
        )
        sol_true = torch.from_numpy(sol_true.astype(np.float32)).to(device)
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            # TODO: all problems
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux,
                optSolver=self.optSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # convert tensor
        cp = coeff_hat.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(
                cp, params, problem, self.optSolver, self.processes, self.pool
            )
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get obj
        expand_shape = torch.Size([solpool.shape[0]] + list(coeff_hat.shape[1:]))
        coeff_hat_pool = coeff_hat.expand(*expand_shape)
        obj_cp = problem.get_objective(coeff_hat, sol_true)
        objpool_cp = problem.get_objective(coeff_hat_pool, solpool)
        # obj_cp = torch.einsum("bd,bd->b", coeff_hat, sol_true).unsqueeze(1)
        # objpool_cp = torch.einsum("bd,nd->bn", coeff_hat, solpool)
        # get loss
        if self.optSolver.modelSense == GRB.MINIMIZE:
            # loss = (obj_cp - objpool_cp).mean(axis=1)
            loss = (obj_cp - objpool_cp).mean(axis=0)
        if self.optSolver.modelSense == GRB.MAXIMIZE:
            # loss = (objpool_cp - obj_cp).mean(axis=1)
            loss = (objpool_cp - obj_cp).mean(axis=0)
        # reduction
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            pass
        else:
            raise ValueError("No reduction '{}'.".format(hyperparams["reduction"]))
        return loss


# class contrastiveMAP(optModel):
#     """
#     An autograd module for Maximum A Posterior contrastive estimation as
#     surrogate loss functions, which is a efficient self-contrastive algorithm.

#     For the MAP, the cost vector needs to be predicted from contextual data and
#     maximizes the separation of the probability of the optimal solution.

#     Thus, allows us to design an algorithm based on stochastic gradient descent.

#     Reference: <https://www.ijcai.org/proceedings/2021/390>
#     """

#     def __init__(self, optSolver, processes=1, solve_ratio=1):
#         """
#         Args:
#             optSolver (optModel): an  optimization model
#             processes (int): number of processors, 1 for single-core, 0 for all of cores
#             solve_ratio (float): the ratio of new solutions computed during training
#         """
#         super().__init__(optSolver, processes, solve_ratio)
#         # solution pool
#         self.solpool = np.unique(dataset.sols.copy(), axis=0)  # remove duplicate

#     def forward(self, coeff_hat, sol_true, reduction="mean"):
#         """
#         Forward pass
#         """
#         # get device
#         device = coeff_hat.device
#         # convert tensor
#         cp = coeff_hat.detach().to("cpu").numpy()
#         # solve
#         if np.random.uniform() <= self.solve_ratio:
#             sol, _ = _solve_in_pass(cp, self.optSolver, self.processes, self.pool)
#             # add into solpool
#             self.solpool = np.concatenate((self.solpool, sol))
#             # remove duplicate
#             self.solpool = np.unique(self.solpool, axis=0)
#         solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
#         # get current obj
#         obj_cp = torch.einsum("bd,bd->b", coeff_hat, sol_true).unsqueeze(1)
#         # get obj for solpool
#         objpool_cp = torch.einsum("bd,nd->bn", coeff_hat, solpool)
#         # get loss
#         if self.optSolver.modelSense == GRB.MINIMIZE:
#             loss, _ = (obj_cp - objpool_cp).max(axis=1)
#         if self.optSolver.modelSense == GRB.MAXIMIZE:
#             loss, _ = (objpool_cp - obj_cp).max(axis=1)
#         # reduction
#         if hyperparams["reduction"] == "mean":
#             loss = torch.mean(loss)
#         elif hyperparams["reduction"] == "sum":
#             loss = torch.sum(loss)
#         elif hyperparams["reduction"] == "none":
#             pass
#         else:
#             raise ValueError("No reduction '{}'.".format(reduction))
#         return loss
