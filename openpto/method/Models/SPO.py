#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Solvers.utils_solver import _solve_in_pass  # , _cache_in_pass


class SPO(optModel):
    """
    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, is_plus=True, **kwargs):
        """
        Args:
            optSolver (optSolver): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)
        # build carterion
        if is_plus:
            self.spo_func = SPOPlusFunc()
        else:
            self.spo_func = SPOFunc()

    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        """
        Forward pass
        """
        if coeff_hat.dim() == 1:
            coeff_hat, coeff_true = coeff_hat.unsqueeze(0), coeff_true.unsqueeze(0)
        sol_true, obj_true = problem.get_decision(
            coeff_true.cpu(),
            params,
            isTrain=False,
            optSolver=self.optSolver,
            **problem.init_API(),
        )
        #
        loss = self.spo_func.apply(
            coeff_hat,
            coeff_true,
            sol_true,
            obj_true,
            problem,
            params,
            self.optSolver,
            self.processes,
            self.pool,
            self.solve_ratio,
            self,
        )
        # reduction
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            loss = loss
        else:
            raise ValueError("No reduction {}".format(hyperparams["reduction"]))
        return loss


class SPOFunc(torch.autograd.Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        coeff_true,
        sol_true,
        obj_true,
        problem,
        params,
        optSolver,
        processes,
        pool,
        solve_ratio,
        module,
    ):
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
            torch.tensor: SPO loss
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        cp = coeff_hat.detach().to("cpu").numpy()
        coeff_true.detach().to("cpu").numpy()
        # check sol
        # _check_sol(c, w, z)
        # solve
        if np.random.uniform() <= solve_ratio:
            sol_hat, obj_hat = _solve_in_pass(
                cp, params, problem, optSolver, processes, pool
            )
            if solve_ratio < 1:
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sol_hat))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            raise NotImplementedError
            # sol, obj = _cache_in_pass(2*cp-c, optSolver, module.solpool)
        # calculate loss
        loss = []
        # TODO: check sign of the loss
        for i in range(len(cp)):
            obj_hat = problem.get_objective(cp[[i]], sol_hat[[i]])
            if torch.is_tensor(obj_hat):
                obj_hat = obj_hat.cpu().numpy()
            instance_loss = -obj_true[[i]] + obj_hat
            loss.append(instance_loss)
            # loss.append(-obj[i] + np.dot(cp[i], w[i]) - z[i])
        # sense
        if optSolver.modelSense == GRB.MINIMIZE:
            loss = np.array(loss)
        if optSolver.modelSense == GRB.MAXIMIZE:
            loss = -np.array(loss)
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = torch.FloatTensor(sol_hat).to(device)
        sol_true = torch.FloatTensor(sol_true.float()).to(device)
        # save solutions
        ctx.save_for_backward(sol_true, sol)
        # add other objects to ctx
        ctx.cp = cp
        ctx.modelSense = optSolver.modelSense
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO
        """
        w, wq = ctx.saved_tensors
        if ctx.modelSense == GRB.MINIMIZE:
            grad = w - wq
        elif ctx.modelSense == GRB.MAXIMIZE:
            grad = wq - w
        else:
            assert 0
        ##### work around #####
        cp = ctx.cp
        if grad.shape != cp.shape:
            if np.prod(grad.shape) == np.prod(cp.shape):
                grad = grad.reshape(cp.shape)
            else:
                grad_shape = grad.shape
                grad = grad.view(*grad_shape, 1).expand(*grad_shape, cp.shape[-1])
        ##### end #####
        return (
            grad_output * grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SPOPlusFunc(torch.autograd.Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        coeff_true,
        sol_true,
        obj_true,
        problem,
        params,
        optSolver,
        processes,
        pool,
        solve_ratio,
        module,
    ):
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
        # check sol
        # _check_sol(c, w, z)
        # solve
        if np.random.uniform() <= solve_ratio:
            sol, obj = _solve_in_pass(
                2 * cp - c, params, problem, optSolver, processes, pool
            )
            if solve_ratio < 1:
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sol))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            raise NotImplementedError
            # sol, obj = _cache_in_pass(2*cp-c, optSolver, module.solpool)
        # calculate loss
        loss = -obj + 2 * problem.get_objective(cp, sol) - obj_true
        # loss = []
        # for i in range(len(cp)):
        #     loss.append(-obj[[i]] + 2 * problem.get_objective(cp[[i]], sol[[i]]) - obj_true[[i]])
        #     loss.append(-obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
        # convert to tensor
        if not torch.is_tensor(loss):
            loss = torch.from_numpy(np.array(loss))
        loss = loss.to(device)
        # sense
        if optSolver.modelSense == GRB.MINIMIZE:
            loss = loss
        if optSolver.modelSense == GRB.MAXIMIZE:
            loss = -loss

        sol = np.array(sol)
        sol = torch.FloatTensor(sol).to(device)
        sol_true = torch.FloatTensor(sol_true).to(device)
        # save solutions
        ctx.save_for_backward(sol_true, sol)
        # add other objects to ctx
        ctx.modelSense = optSolver.modelSense
        ctx.cp = cp
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        if ctx.modelSense == GRB.MINIMIZE:
            grad = 2 * (w - wq)
        if ctx.modelSense == GRB.MAXIMIZE:
            grad = 2 * (wq - w)
        ##### work around #####
        cp = ctx.cp
        if grad.shape != cp.shape:
            if np.prod(grad.shape) == np.prod(cp.shape):
                grad = grad.reshape(cp.shape)
            else:
                grad_shape = grad.shape
                grad = grad.view(*grad_shape, 1).expand(*grad_shape, cp.shape[-1])
        ##### end #####
        return (
            grad_output * grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
