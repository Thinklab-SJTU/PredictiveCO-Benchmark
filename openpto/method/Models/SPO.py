#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import to_tensor


class SPO(optModel):
    """
    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optSolver): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)
        self.spo_func = SPOPlusFunc()

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
        sols_true, objs_true = problem.get_decision(
            coeff_true,
            params,
            isTrain=False,
            optSolver=self.optSolver,
            **problem.init_API(),
        )
        #
        loss = self.spo_func.apply(
            coeff_hat, coeff_true, sols_true, objs_true, problem, params, self.optSolver
        )
        # reduction
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            pass
        else:
            raise ValueError("No reduction {}".format(hyperparams["reduction"]))
        return loss


class SPOPlusFunc(torch.autograd.Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        coeff_true,
        sols_true,
        objs_true,
        problem,
        params,
        optSolver,
    ):
        """
        Forward pass for SPO+

        Args:
            coeff_hat (torch.tensor): a batch of predicted values of the cost
            coeff_true (torch.tensor): a batch of true values of the cost
            sols_true (torch.tensor): a batch of true optimal solutions
            objs_true (torch.tensor): a batch of true optimal objective values
            problem: a problem object
            params: a parameter object
            optSolver (optSolver): an optimization solver

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        coeff_hat_cpu = coeff_hat.detach().cpu()
        coeff_true_cpu = coeff_true.detach().cpu()
        # solve
        sols_proxy, obj_proxy = problem.get_decision(
            2 * coeff_hat_cpu - coeff_true_cpu,
            params,
            optSolver,
            **problem.init_API(),
        )
        # calculate loss
        loss = (
            -to_tensor(obj_proxy).cpu()
            + 2 * to_tensor(problem.get_objective(coeff_hat_cpu, sols_true)).cpu()
            - to_tensor(objs_true).cpu()
        )
        # convert to tensor
        loss = to_tensor(loss).to(device)
        sols_proxy = to_tensor(sols_proxy).to(device)
        sols_true = to_tensor(sols_true).to(device)
        # save solutions
        ctx.save_for_backward(sols_true, sols_proxy)
        # add other objects to ctx
        ctx.modelSense = optSolver.modelSense
        ctx.coeff_hat_cpu = coeff_hat_cpu
        # model sense
        if optSolver.modelSense == GRB.MINIMIZE:
            pass
        elif optSolver.modelSense == GRB.MAXIMIZE:
            loss = -loss
        else:
            raise NotImplementedError
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        sols_true, sols_proxy = ctx.saved_tensors
        if ctx.modelSense == GRB.MINIMIZE:
            grad = 2 * (sols_true - sols_proxy)
        elif ctx.modelSense == GRB.MAXIMIZE:
            grad = -2 * (sols_true - sols_proxy)
        ##### work around #####
        coeff_hat_cpu = ctx.coeff_hat_cpu
        if grad.shape != coeff_hat_cpu.shape:
            if np.prod(grad.shape) == np.prod(coeff_hat_cpu.shape):
                grad = grad.reshape(coeff_hat_cpu.shape)
            else:
                grad_shape = grad.shape
                grad = grad.view(*grad_shape, 1).expand(
                    *grad_shape, coeff_hat_cpu.shape[-1]
                )
        ##### end #####
        return (grad_output * grad, None, None, None, None, None, None)
