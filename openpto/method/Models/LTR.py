#!/usr/bin/env python
# coding: utf-8
"""
Learning to rank Losses
"""

import numpy as np
import torch
import torch.nn.functional as F

from gurobipy import GRB  # pylint: disable=no-name-in-module
from torch import nn

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Solvers.utils_solver import _solve_in_pass


# TODO: currently only support single-instance batch
class listwiseLTR(optModel):
    """
    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    Code from: https://github.com/khalil-research/PyEPO/blob/NCE/pkg/pyepo/func/rank.py
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)
        # solution pool
        n_vars = optSolver.num_vars
        self.solpool = np.empty((0, n_vars))

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            # TODO: all problems
            _, Y_train, _ = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=params,
                optSolver=self.optSolver,
                isTrain=False,
                **problem.init_API(),
            )
            print("self.solpool: ", self.solpool.shape)
        # convert tensor
        cp = coeff_hat.detach().to("cpu").numpy()
        # solve
        if True:
            # if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(
                cp, params, problem, self.optSolver, self.processes, self.pool
            )
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool
        # TODO: currently only support linear objective
        objpool_c = problem.get_objective(coeff_true, solpool.unsqueeze(-1))
        objpool_cp = problem.get_objective(coeff_hat, solpool.unsqueeze(-1))
        # objpool_c = coeff_true @ solpool.T  # true cost
        # objpool_cp = coeff_hat @ solpool.T  # pred cost
        # cross entropy loss
        if self.optSolver.modelSense == GRB.MINIMIZE:
            # loss = -(F.log_softmax(objpool_cp, dim=1) * F.softmax(objpool_c, dim=1))
            loss = -(F.log_softmax(objpool_cp, dim=0) * F.softmax(objpool_c, dim=0))
        if self.optSolver.modelSense == GRB.MAXIMIZE:
            # loss = -(F.log_softmax(-objpool_cp, dim=1) * F.softmax(-objpool_c, dim=1))
            loss = -(F.log_softmax(-objpool_cp, dim=0) * F.softmax(-objpool_c, dim=0))
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

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)
        # solution pool
        n_vars = optSolver.num_vars
        self.solpool = np.empty((0, n_vars))

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            # TODO: all problems
            _, Y_train, _ = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=params,
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
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool
        objpool_c = problem.get_objective(coeff_true, solpool.unsqueeze(-1))
        objpool_cp = problem.get_objective(coeff_hat, solpool.unsqueeze(-1))
        # objpool_c = torch.einsum("bd,nd->bn", coeff_true, solpool)  # true cost
        # objpool_cp = torch.einsum("bd,nd->bn", coeff_hat, solpool)  # pred cost
        # init relu as max(0,x)
        relu = nn.ReLU()
        # init loss
        loss = []
        for i in range(len(coeff_hat)):
            # best sol
            if self.optSolver.modelSense == GRB.MINIMIZE:
                best_ind = torch.argmin(objpool_c[i])
                # best_ind = torch.argmin(objpool_c)
            if self.optSolver.modelSense == GRB.MAXIMIZE:
                best_ind = torch.argmax(objpool_c[i])
                # best_ind = torch.argmax(objpool_c)
            objpool_cp_best = objpool_cp[i, best_ind]
            # objpool_cp_best = objpool_cp[best_ind]
            # rest sol
            rest_ind = [j for j in range(len(objpool_cp[i])) if j != best_ind]
            objpool_cp_rest = objpool_cp[i, rest_ind]
            # rest_ind = [j for j in range(len(objpool_cp)) if j != best_ind]
            # objpool_cp_rest = objpool_cp[rest_ind]
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
    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)
        # solution pool
        n_vars = optSolver.num_vars
        self.solpool = np.empty((0, n_vars))

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # get device
        device = coeff_hat.device
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            # TODO: all problems
            _, Y_train, _ = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=params,
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
            # print("shape: ", self.solpool.shape, sol.shape)
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool as score
        objpool_c = problem.get_objective(coeff_true, solpool.unsqueeze(-1))
        objpool_cp = problem.get_objective(coeff_hat, solpool.unsqueeze(-1))
        # objpool_c = coeff_true @ solpool.T  # true cost
        # objpool_cp = coeff_hat @ solpool.T  # pred cost
        # squared loss
        # loss = (objpool_c - objpool_cp).square().mean(axis=1)
        loss = (objpool_c - objpool_cp).square().mean(axis=0)
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
