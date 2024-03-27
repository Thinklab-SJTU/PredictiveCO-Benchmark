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
from openpto.method.utils_method import do_reduction, to_tensor


class pointwiseLTR(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        # solution pool
        n_vars = ptoSolver.num_vars
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
                ptoSolver=self.ptoSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # solve
        sol_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, self.ptoSolver, **problem.init_API()
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
        objpool_c = problem.get_objective(coeff_true, solpool, params)
        objpool_c_hat = problem.get_objective(coeff_hat, solpool, params)
        # squared loss
        loss = (objpool_c - objpool_c_hat).square().mean(axis=0)
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss


class pairwiseLTR(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        # solution pool
        n_vars = ptoSolver.num_vars
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
                ptoSolver=self.ptoSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # solve
        sol_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, self.ptoSolver, **problem.init_API()
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
        objpool_c_true = problem.get_objective(coeff_true_pool, solpool, params)
        objpool_c_hat_pool = problem.get_objective(coeff_hat_pool, solpool, params)
        # TODO: currently, only support batch-1 training
        # init loss
        loss = []
        for i in range(len(coeff_hat)):
            # best sol
            if self.ptoSolver.modelSense == GRB.MINIMIZE:
                # best_ind = torch.argmin(objpool_c_true[i])
                best_ind = torch.argmin(objpool_c_true)
            elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
                # best_ind = torch.argmax(objpool_c_true[i])
                best_ind = torch.argmax(objpool_c_true)
            else:
                raise NotImplementedError
            objpool_cp_best = objpool_c_hat_pool[best_ind]
            # rest sol
            rest_ind = [j for j in range(len(objpool_c_hat_pool)) if j != best_ind]
            objpool_cp_rest = objpool_c_hat_pool[rest_ind]
            # best vs rest loss
            if self.ptoSolver.modelSense == GRB.MINIMIZE:
                loss.append(F.relu(objpool_cp_best - objpool_cp_rest))
            elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
                loss.append(F.relu(objpool_cp_rest - objpool_cp_best))
            else:
                raise NotImplementedError
        loss = torch.stack(loss)
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss


class listwiseLTR(optModel):
    """
    Reference:
    Code from:
    """

    def __init__(self, ptoSolver, tau=1.0, **kwargs):
        """ """
        super().__init__(ptoSolver)

        if tau <= 0:
            raise ValueError("tau is not positive.")
        self.tau = tau
        # solution pool
        n_vars = ptoSolver.num_vars
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
                ptoSolver=self.ptoSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # solve #TODO: if sol pool reasonable?
        sol_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, self.ptoSolver, **problem.init_API()
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
        objpool_c = problem.get_objective(coeff_true, solpool, params)
        objpool_c_hat = problem.get_objective(coeff_hat, solpool, params)
        # cross entropy loss
        if self.ptoSolver.modelSense == GRB.MINIMIZE:
            loss = -(
                F.log_softmax(-objpool_c_hat / self.tau, dim=0)
                * F.softmax(-objpool_c / self.tau, dim=0)
            )
        elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
            loss = -(F.log_softmax(objpool_c_hat, dim=0) * F.softmax(objpool_c, dim=0))
        else:
            raise NotImplementedError
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss
