#!/usr/bin/env python
# coding: utf-8
"""
Learning to rank Losses
"""

import numpy as np
import torch
import torch.nn.functional as F

from gurobipy import GRB
from torch import nn

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Solvers.utils_solver import _solve_in_pass
from openpto.problems.PTOProblem import PTOProblem


class listwiseLTR(optModel):
    """
    An autograd module for listwise learning to rank, where the goal is to learn
    an objective function that ranks a pool of feasible solutions correctly.

    For the listwise LTR, the cost vector needs to be predicted from contextual
    data, and the loss measures the scores of the whole ranked lists.

    Thus, allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optSolver (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (PTOProblem): the training data, usually this is simply the training set
        """
        super().__init__(optSolver, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, PTOProblem):  # type checking
            raise TypeError("dataset is not an PTOProblem")
        self.solpool = np.unique(dataset.sols.copy(), axis=0)  # remove duplicate

    def forward(self, coeff_hat, coeff_true, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # convert tensor
        cp = coeff_hat.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optSolver, self.processes, self.pool)
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool
        objpool_c = coeff_true @ solpool.T  # true cost
        objpool_cp = coeff_hat @ solpool.T  # pred cost
        # cross entropy loss
        if self.optSolver.modelSense == GRB.MINIMIZE:
            loss = -(F.log_softmax(objpool_cp, dim=1) * F.softmax(objpool_c, dim=1))
        if self.optSolver.modelSense == GRB.MAXIMIZE:
            loss = -(F.log_softmax(-objpool_cp, dim=1) * F.softmax(-objpool_c, dim=1))
        # reduction
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(hyperparams["reduction"]))
        return loss


class pairwiseLTR(optModel):
    """
    An autograd module for pairwise learning to rank, where the goal is to learn
    an objective function that ranks a pool of feasible solutions correctly.

    For the pairwise LTR, the cost vector needs to be predicted from contextual
    data, and the loss learns the relative ordering of pairs of items.

    Thus, allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optSolver (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (PTOProblem): the training data
        """
        super().__init__(optSolver, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, PTOProblem):  # type checking
            raise TypeError("dataset is not an PTOProblem")
        self.solpool = np.unique(dataset.sols.copy(), axis=0)  # remove duplicate

    def forward(self, coeff_hat, coeff_true, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # convert tensor
        cp = coeff_hat.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optSolver, self.processes, self.pool)
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool
        objpool_c = torch.einsum("bd,nd->bn", coeff_true, solpool)  # true cost
        objpool_cp = torch.einsum("bd,nd->bn", coeff_hat, solpool)  # pred cost
        # init relu as max(0,x)
        relu = nn.ReLU()
        # init loss
        loss = []
        for i in range(len(coeff_hat)):
            # best sol
            if self.optSolver.modelSense == GRB.MINIMIZE:
                best_ind = torch.argmin(objpool_c[i])
            if self.optSolver.modelSense == GRB.MAXIMIZE:
                best_ind = torch.argmax(objpool_c[i])
            objpool_cp_best = objpool_cp[i, best_ind]
            # rest sol
            rest_ind = [j for j in range(len(objpool_cp[i])) if j != best_ind]
            objpool_cp_rest = objpool_cp[i, rest_ind]
            # best vs rest loss
            if self.optSolver.modelSense == GRB.MINIMIZE:
                loss.append(relu(objpool_cp_best - objpool_cp_rest).mean())
            if self.optSolver.modelSense == GRB.MAXIMIZE:
                loss.append(relu(objpool_cp_rest - objpool_cp_best).mean())
        loss = torch.stack(loss)
        # reduction
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(hyperparams["reduction"]))
        return loss


class pointwiseLTR(optModel):
    """
    An autograd module for pointwise learning to rank, where the goal is to
    learn an objective function that ranks a pool of feasible solutions
    correctly.

    For the pointwise LTR, the cost vector needs to be predicted from contextual
    data, and calculates the ranking scores of the items.

    Thus, allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optSolver (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (PTOProblem): the training data
        """
        super().__init__(optSolver, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, PTOProblem):  # type checking
            raise TypeError("dataset is not an PTOProblem")
        self.solpool = np.unique(dataset.sols.copy(), axis=0)  # remove duplicate

    def forward(self, coeff_hat, coeff_true, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # convert tensor
        cp = coeff_hat.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optSolver, self.processes, self.pool)
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool as score
        objpool_c = coeff_true @ solpool.T  # true cost
        objpool_cp = coeff_hat @ solpool.T  # pred cost
        # squared loss
        loss = (objpool_c - objpool_cp).square().mean(axis=1)
        # reduction
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(hyperparams["reduction"]))
        return loss
