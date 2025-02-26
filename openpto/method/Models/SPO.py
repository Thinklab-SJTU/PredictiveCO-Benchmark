#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_tensor


class SPO(optModel):
    """ """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
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
            ptoSolver=self.ptoSolver,
            **problem.init_API(),
        )
        #
        loss = self.spo_func.apply(
            coeff_hat, coeff_true, sols_true, objs_true, problem, params, self.ptoSolver
        )
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss


class SPOPlusFunc(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        coeff_true,
        sols_true,
        objs_true,
        problem,
        params,
        ptoSolver,
    ):
        """ """
        # get device
        device = coeff_hat.device
        # convert tenstor
        coeff_hat_cpu = coeff_hat.detach().cpu()
        coeff_true_cpu = coeff_true.detach().cpu()
        # solve
        sols_proxy, obj_proxy = problem.get_decision(
            2 * coeff_hat_cpu - coeff_true_cpu,
            params,
            ptoSolver,
            **problem.init_API(),
        )
        dq_hat = problem.get_objective(coeff_hat_cpu, sols_true, params)
        # calculate loss
        loss = (
            -to_tensor(obj_proxy).cpu()
            + 2 * to_tensor(dq_hat).cpu()
            - to_tensor(objs_true).cpu()
        )
        # convert to tensor
        loss = to_tensor(loss).to(device)
        sols_proxy = to_tensor(sols_proxy).to(device)
        sols_true = to_tensor(sols_true).to(device)
        # save solutions
        ctx.save_for_backward(sols_true, sols_proxy)
        # add other objects to ctx
        ctx.modelSense = ptoSolver.modelSense
        ctx.coeff_hat_cpu = coeff_hat_cpu
        # model sense
        if ptoSolver.modelSense == GRB.MINIMIZE:
            pass
        elif ptoSolver.modelSense == GRB.MAXIMIZE:
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
        # print("1:: coeff_hat_cpu, grad_output, grad: ", coeff_hat_cpu.shape,  grad_output.shape, grad.shape)
        if grad.shape != coeff_hat_cpu.shape:
            if np.prod(grad.shape) == np.prod(coeff_hat_cpu.shape):
                grad = grad.reshape(coeff_hat_cpu.shape)
            else:
                # expand grad and grad_output shapes, to the shape of coeff_hat_cpu
                if grad.ndim < coeff_hat_cpu.ndim:
                    grad_shape = grad.shape
                    grad = grad.view(*grad_shape, 1).expand(
                        *grad_shape, coeff_hat_cpu.shape[-1]
                    )
                if grad_output.ndim < coeff_hat_cpu.ndim:
                    new_shape = grad_output.shape + (1,) * (
                        coeff_hat_cpu.ndim - grad_output.ndim
                    )
                    grad_output = grad_output.view(new_shape)
                    grad_output = grad_output.expand(coeff_hat_cpu.shape)
                # print("2:: grad_output, grad: ", coeff_hat_cpu.shape,  grad_output.shape, grad.shape)
                # ## when a batch contains multiple items, do:
                # if grad_output.ndim < grad.ndim:
                #     grad_output_shape = grad_output.shape
                #     grad_output = grad_output.unsqueeze(1)
                #     grad_output = grad_output.view(*grad_output_shape, 1).expand(
                #         grad_output_shape[0],
                #         coeff_hat_cpu.shape[1],
                #         *grad_output_shape[1:],
                #     )
        # print("3:: grad_output, grad: ", grad_output.shape, grad.shape)
        ##### end #####
        return (grad_output * grad, None, None, None, None, None, None)
