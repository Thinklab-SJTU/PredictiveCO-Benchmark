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
from openpto.method.utils_method import to_tensor


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
        self.solpool = np.empty((0, n_vars), dtype=np.float32)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """

        # coeff_hat = coeff_hat.squeeze(-1)
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux[:1],
                optSolver=self.optSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # convert tensor
        coeff_hat_array = coeff_hat.detach().cpu().numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol_hat, _ = problem.get_decision(
                coeff_hat_array, params, self.optSolver, **problem.init_API()
            )
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol_hat))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = to_tensor(self.solpool).to(coeff_hat.device)
        # obj for solpool as score
        expand_shape = torch.Size([solpool.shape[0]] + list(coeff_hat.shape[1:]))
        coeff_hat = coeff_hat.expand(*expand_shape)
        coeff_true = coeff_true.expand(*expand_shape)
        #
        objpool_c = problem.get_objective(coeff_true, solpool)
        objpool_c_hat = problem.get_objective(coeff_hat, solpool)
        # squared loss
        loss = (objpool_c - objpool_c_hat).square().mean(axis=0)
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
        self.solpool = np.empty((0, n_vars), dtype=np.float32)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux,
                optSolver=self.optSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # convert tensor
        coeff_hat_array = coeff_hat.detach().cpu().numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol_hat, _ = problem.get_decision(
                coeff_hat_array, params, self.optSolver, **problem.init_API()
            )
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol_hat))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        solpool = to_tensor(self.solpool).to(coeff_hat.device)
        # transform to tensor
        expand_shape = torch.Size([solpool.shape[0]] + list(coeff_hat.shape[1:]))
        coeff_hat_pool = coeff_hat.expand(*expand_shape)
        coeff_true_pool = coeff_true.expand(*expand_shape)
        # obj for solpool
        objpool_c_true = problem.get_objective(coeff_true_pool, solpool)
        objpool_c_hat_pool = problem.get_objective(coeff_hat_pool, solpool)
        # TODO: currently, only support batch-1 training
        # init loss
        loss = []
        for i in range(len(coeff_hat)):
            # best sol
            if self.optSolver.modelSense == GRB.MINIMIZE:
                # best_ind = torch.argmin(objpool_c_true[i])
                best_ind = torch.argmin(objpool_c_true)
            elif self.optSolver.modelSense == GRB.MAXIMIZE:
                # best_ind = torch.argmax(objpool_c_true[i])
                best_ind = torch.argmax(objpool_c_true)
            else:
                raise NotImplementedError
            objpool_cp_best = objpool_c_hat_pool[best_ind]
            # objpool_cp_best = objpool_c_hat_pool[i, best_ind]
            # rest sol
            rest_ind = [j for j in range(len(objpool_c_hat_pool)) if j != best_ind]
            # rest_ind = [j for j in range(len(objpool_c_hat_pool[i])) if j != best_ind]
            objpool_cp_rest = objpool_c_hat_pool[rest_ind]
            # objpool_cp_rest = objpool_c_hat_pool[i, rest_ind]
            # best vs rest loss
            if self.optSolver.modelSense == GRB.MINIMIZE:
                loss.append(F.relu(objpool_cp_best - objpool_cp_rest))
            elif self.optSolver.modelSense == GRB.MAXIMIZE:
                loss.append(F.relu(objpool_cp_rest - objpool_cp_best))
            else:
                raise NotImplementedError
        loss = torch.stack(loss)
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


class listwiseLTR(optModel):
    """
    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    Code from: https://github.com/khalil-research/PyEPO/blob/NCE/pkg/pyepo/func/rank.py
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, tau=1.0, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)

        if tau <= 0:
            raise ValueError("tau is not positive.")
        self.tau = tau
        # solution pool
        n_vars = optSolver.num_vars
        self.solpool = np.empty((0, n_vars), dtype=np.float32)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # coeff_hat = coeff_hat.squeeze(-1)
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux,
                optSolver=self.optSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # convert tensor
        coeff_hat_array = coeff_hat.detach().cpu().numpy()
        # solve #TODO: if sol pool reasonable?
        if np.random.uniform() <= self.solve_ratio:
            sol_hat, _ = problem.get_decision(
                coeff_hat_array, params, self.optSolver, **problem.init_API()
            )
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol_hat))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = to_tensor(self.solpool).to(coeff_hat.device)
        expand_shape = torch.Size([solpool.shape[0]] + list(coeff_hat.shape[1:]))
        coeff_hat = coeff_hat.expand(*expand_shape)
        coeff_true = coeff_true.expand(*expand_shape)
        # obj for solpool
        objpool_c = problem.get_objective(coeff_true, solpool)
        objpool_c_hat = problem.get_objective(coeff_hat, solpool)
        # cross entropy loss
        if self.optSolver.modelSense == GRB.MINIMIZE:
            loss = -(
                F.log_softmax(-objpool_c_hat / self.tau, dim=0)
                * F.softmax(-objpool_c / self.tau, dim=0)
            )
        elif self.optSolver.modelSense == GRB.MAXIMIZE:
            loss = -(F.log_softmax(objpool_c_hat, dim=0) * F.softmax(objpool_c, dim=0))
        else:
            raise NotImplementedError
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
