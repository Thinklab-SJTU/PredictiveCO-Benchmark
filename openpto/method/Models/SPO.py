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
            raise NotImplementedError
            # self.spo_func = SPOFunc()

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
            coeff_true.cpu(),
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
        sol_proxy, obj_proxy = problem.get_decision(
            2 * coeff_hat_array - coeff_true_array,
            params,
            optSolver,
            **problem.init_API(),
        )
        # calculate loss
        loss = (
            -obj_proxy + 2 * problem.get_objective(coeff_hat_array, sols_true) - objs_true
        )
        # convert to tensor
        if not torch.is_tensor(loss):
            loss = torch.from_numpy(loss)
        loss = loss.to(device)
        # save solutions
        if not torch.is_tensor(sol_proxy):
            sol_proxy = torch.from_numpy(sol_proxy)
        sol_proxy = sol_proxy.to(device)
        if not torch.is_tensor(sols_true):
            sols_true = torch.from_numpy(sols_true)
        sols_true = sols_true.to(device)
        ctx.save_for_backward(sols_true, sol_proxy)
        # add other objects to ctx
        ctx.modelSense = optSolver.modelSense
        ctx.coeff_hat_array = coeff_hat_array
        # TODO: check
        # sense
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
        sols_true, sol_proxy = ctx.saved_tensors
        grad = 2 * (sols_true - sol_proxy)
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


# class SPOFunc(torch.autograd.Function):
#     """
#     A autograd function for SPO+ Loss
#     """

#     @staticmethod
#     def forward(
#         ctx, coeff_hat, coeff_true, sols_true, objs_true, problem, params, optSolver
#     ):
#         """
#         Forward pass for SPO+

#         Args:
#             coeff_hat (torch.tensor): a batch of predicted values of the cost
#             coeff_true (torch.tensor): a batch of true values of the cost
#             sols_true (torch.tensor): a batch of true optimal solutions
#             objs_true (torch.tensor): a batch of true optimal objective values
#             optSolver (optSolver): an optimization solver
#             processes (int): number of processors, 1 for single-core, 0 for all of cores
#             pool (ProcessPool): process pool object
#             solve_ratio (float): the ratio of new solutions computed during training
#             module (optModule): SPOPlus modeul

#         Returns:
#             torch.tensor: SPO loss
#         """
#         # get device
#         device = coeff_hat.device
#         # convert tenstor
#         coeff_hat_array = coeff_hat.detach().to("cpu").numpy()
#         # solve
#         sols_hat, _ = problem.get_decision(
#             coeff_hat_array, params, optSolver, **problem.init_API()
#         )
#         # calculate loss
#         obj2 = problem.get_objective(coeff_true, sols_hat)
#         # TODO: check sign of the loss
#         loss = -objs_true + obj2
#         # sense
#         if optSolver.modelSense == GRB.MINIMIZE:
#             pass
#         elif optSolver.modelSense == GRB.MAXIMIZE:
#             loss = -loss
#         # convert to tensor
#         sols_hat = torch.from_numpy(sols_hat).to(device)
#         sols_true = torch.from_numpy(sols_true.float()).to(device)
#         # save solutions
#         ctx.save_for_backward(sols_hat, sols_true)
#         # add other objects to ctx
#         ctx.coeff_hat_array = coeff_hat_array
#         ctx.modelSense = optSolver.modelSense
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass for SPO
#         """
#         sols_hat, sols_true = ctx.saved_tensors
#         grad = sols_hat  # TODO
#         ##### work around #####
#         coeff_hat_array = ctx.coeff_hat_array
#         if grad.shape != coeff_hat_array.shape:
#             if np.prod(grad.shape) == np.prod(coeff_hat_array.shape):
#                 grad = grad.reshape(coeff_hat_array.shape)
#             else:
#                 grad_shape = grad.shape
#                 grad = grad.view(*grad_shape, 1).expand(
#                     *grad_shape, coeff_hat_array.shape[-1]
#                 )
#         ##### end #####
#         return (grad_output * grad, None, None, None, None, None, None)
