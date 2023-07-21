#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
"""

import numpy as np
import torch

from gurobipy import GRB
# from pyepo.func.abcmodule import optModule
# from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass
from openpto.method.models.abcOptModel import optModel
from openpto.method.Solver.utils_solver import _solve_in_pass#, _cache_in_pass

class SPOPlus(optModel):
    """
    An autograd module for SPO+ Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector need to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1):
        """
        Args:
            optSolver (optSolver): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optSolver, processes, solve_ratio)
        # build carterion
        self.spop = SPOPlusFunc()
    
    def forward(self, problem, coeff_hat, coeff_true=None, sol_hat=None, sol_true=None, params=None, **hyperparams):
    # def forward(self, coeff_hat, coeff_true, sol_true, obj_true, reduction="mean"):
        """
        Forward pass
        """
        obj_true = None
        if sol_true is None:
            if hasattr(problem, 'get_decision_and_objective'):
                print("line 51: ",coeff_true, params)
                sol_true, obj_true = problem.get_decision_and_objective(coeff_true.cpu(), params.cpu(), 
                                            isTrain = False, optSolver=None, **problem.params_API())
            else:
                sol_true = problem.get_decision(coeff_true.cpu().numpy(), params.cpu().numpy(), 
                                                isTrain = False, optSolver=None, **problem.params_API())
        if obj_true is None:
            obj_true = problem.get_objective(coeff_true, sol_true)
        loss = self.spop.apply(coeff_hat, coeff_true, sol_true, obj_true,
                               self.optSolver, self.processes, self.pool,
                               self.solve_ratio, self)
        # reduction
        if hyperparams['reduction'] == "mean":
            loss = torch.mean(loss)
        elif hyperparams['reduction'] == "sum":
            loss = torch.sum(loss)
        elif hyperparams['reduction'] == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(hyperparams['reduction']))
        return loss


class SPOPlusFunc(torch.autograd.Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, coeff_hat, coeff_true, sol_true, obj_true,
                optSolver, processes, pool, solve_ratio, module):
        """
        Forward pass for SPO+

        Args:
            coeff_hat (torch.tensor): a batch of predicted values of the cost
            coeff_true (torch.tensor): a batch of true values of the cost
            sol_true (torch.tensor): a batch of true optimal solutions
            obj_true (torch.tensor): a batch of true optimal objective values
            optSolver (optSolver): an optimization solver
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): SPOPlus modeul

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        cp = coeff_hat.detach().to("cpu").numpy()
        c = coeff_true.detach().to("cpu").numpy()
        w = sol_true#.detach().to("cpu").numpy()
        z = obj_true#.detach().to("cpu").numpy()
        # check sol
        #_check_sol(c, w, z)
        # solve
        if np.random.uniform() <= solve_ratio:
            sol, obj = _solve_in_pass(2*cp-c, optSolver, processes, pool)
            if solve_ratio < 1:
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sol))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            raise NotImplementedError
            # sol, obj = _cache_in_pass(2*cp-c, optSolver, module.solpool)
        # calculate loss
        loss = []
        for i in range(len(cp)):
            loss.append(- obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
        # sense
        if optSolver.modelSense == GRB.MINIMIZE:
            loss = np.array(loss)
        if optSolver.modelSense == GRB.MAXIMIZE:
            loss = - np.array(loss)
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = np.array(sol)
        sol = torch.FloatTensor(sol).to(device)
        # save solutions
        ctx.save_for_backward(sol_true, sol)
        # add other objects to ctx
        ctx.optSolver = optSolver
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        optSolver = ctx.optSolver
        if optSolver.modelSense == GRB.MINIMIZE:
            grad = 2 * (w - wq)
        if optSolver.modelSense == GRB.MAXIMIZE:
            grad = 2 * (wq - w)
        return grad_output * grad, None, None, None, None, None, None, None, None
