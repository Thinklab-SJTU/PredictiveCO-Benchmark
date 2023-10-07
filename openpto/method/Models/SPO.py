#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel


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
        coeff_hat_array = coeff_hat.detach().to("cpu").numpy()
        coeff_true_array = coeff_true.detach().to("cpu").numpy()
        # solve
        sols_proxy, obj_proxy = problem.get_decision(
            2 * coeff_hat_array - coeff_true_array,
            params,
            optSolver,
            **problem.init_API(),
        )
        # calculate loss
        loss = (
            -obj_proxy
            + 2 * problem.get_objective(coeff_hat_array, sols_true)
            - objs_true.cpu().numpy()
        )
        # convert to tensor
        if not torch.is_tensor(loss):
            loss = torch.from_numpy(loss)
        loss = loss.to(device)
        # save solutions
        if not torch.is_tensor(sols_proxy):
            sols_proxy = torch.from_numpy(sols_proxy)
        sols_proxy = sols_proxy.to(device)
        if not torch.is_tensor(sols_true):
            sols_true = torch.from_numpy(sols_true)
        sols_true = sols_true.to(device)
        ctx.save_for_backward(sols_true, sols_proxy)
        # add other objects to ctx
        ctx.modelSense = optSolver.modelSense
        ctx.coeff_hat_array = coeff_hat_array
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
        # grad = 2 * (sols_true - sols_proxy)
        if ctx.modelSense == GRB.MINIMIZE:
            grad = 2 * (sols_true - sols_proxy)
        elif ctx.modelSense == GRB.MAXIMIZE:
            grad = -2 * (sols_true - sols_proxy)
        ##### work around #####
        coeff_hat_array = ctx.coeff_hat_array
        if grad.shape != coeff_hat_array.shape:
            if np.prod(grad.shape) == np.prod(coeff_hat_array.shape):
                grad = grad.reshape(coeff_hat_array.shape)
            else:
                grad_shape = grad.shape
                grad = grad.view(*grad_shape, 1).expand(
                    *grad_shape, coeff_hat_array.shape[-1]
                )
        ##### end #####
        return (grad_output * grad, None, None, None, None, None, None)
