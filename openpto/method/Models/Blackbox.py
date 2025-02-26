#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Black-box optimization function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import (
    do_reduction,
    minus,
    to_array,
    to_device,
    to_tensor,
)


class blackboxSolver(optModel):
    """ """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        # smoothing parameter
        if kwargs["lambd"] <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = kwargs["lambd"]
        # build blackbox optimizer
        self.dbb = blackboxFunc()

    def forward(
        self,
        problem,
        coeff_hat,
        params,
        **hyperparams,
    ):
        """
        Forward pass
        """
        sols_hat = self.dbb.apply(
            coeff_hat, problem, params, self.ptoSolver, self.lambd, hyperparams
        )
        # TODO:
        return sols_hat


class subopt_blackbox(optModel):
    """ """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        # smoothing parameter
        if kwargs["lambd"] <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = kwargs["lambd"]
        # build blackbox optimizer
        self.dbb = blackboxFunc()

    def forward(
        self,
        problem,
        coeff_hat,
        params,
        **hyperparams,
    ):
        """
        Forward pass
        """
        sols_hat = self.dbb.apply(
            coeff_hat, problem, params, self.ptoSolver, self.lambd, hyperparams
        )
        objs_hat = problem.get_objective(coeff_hat, sols_hat, params, **hyperparams)

        # reduction
        loss = do_reduction(objs_hat, hyperparams["reduction"])

        if self.ptoSolver.modelSense == GRB.MINIMIZE:
            pass
        elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
            loss = -loss

        if (
            hyperparams["do_debug"]
            and hyperparams["partition"] == "train"
            and loss.requires_grad
        ):
            objs_hat_grad = torch.autograd.grad(loss, objs_hat, retain_graph=True)[0]
            coeff_hat_grad = torch.autograd.grad(loss, coeff_hat, retain_graph=True)[0]
            sols_hat_grad = torch.autograd.grad(loss, sols_hat, retain_graph=True)[0]
            print(
                "dbb grad: ",
                objs_hat_grad.shape,
                coeff_hat_grad.shape,
                sols_hat_grad.shape,
            )
        return loss


class blackboxFunc(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        problem,
        params,
        ptoSolver,
        lambd,
        hyperparams,
    ):
        """ """
        # get device
        device = problem.device
        # convert tenstor
        coeff_hat_array = to_array(coeff_hat)
        sols_hat, _ = problem.get_decision(
            to_device(coeff_hat, "cpu"), params, ptoSolver, **problem.init_API()
        )
        # save to ctx (np.ndarray version)
        ctx.coeff_hat_array = coeff_hat_array
        ctx.sols_hat = sols_hat
        ctx.lambd = lambd
        ctx.ptoSolver = ptoSolver
        ctx.params = params
        ctx.problem = problem
        # convert to tensor
        sols_hat = to_device(to_tensor(sols_hat), device)
        return sols_hat

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DBB
        """
        coeff_hat_array = ctx.coeff_hat_array
        sols_hat = ctx.sols_hat
        lambd = ctx.lambd
        ptoSolver = ctx.ptoSolver
        params = ctx.params
        problem = ctx.problem
        # get device
        device = problem.device
        # convert tenstor
        dl = grad_output.detach().cpu().numpy()
        # perturbed costs
        if dl.shape != coeff_hat_array.shape:
            ##### work around #####
            print("dl.shape: ", dl.shape, "coeff_hat_array.shape:", coeff_hat_array.shape)
            if np.prod(dl.shape) == np.prod(coeff_hat_array.shape):
                dl = dl.reshape(coeff_hat_array.shape)
                cq = coeff_hat_array + lambd * dl
            else:
                if dl.ndim > coeff_hat_array.ndim:
                    cq = coeff_hat_array + lambd * dl.mean(-1, keepdims=True)
                else:
                    cq = coeff_hat_array + lambd * np.expand_dims(dl, axis=-1)
            ##### end #####
        else:
            cq = coeff_hat_array + lambd * dl
        # second np call
        sols_lamb, _ = problem.get_decision(
            to_tensor(cq), params, ptoSolver, **problem.init_API()
        )
        grad = minus(sols_lamb, sols_hat) / lambd
        # convert to tensor
        grad = to_device(to_tensor(grad), device)
        ##### work around #####
        if grad.shape != coeff_hat_array.shape:
            if np.prod(grad.shape) == np.prod(coeff_hat_array.shape):
                grad = grad.reshape(coeff_hat_array.shape)
            else:
                grad_shape = grad.shape
                grad = grad.view(*grad_shape, 1).expand(
                    *grad_shape, coeff_hat_array.shape[-1]
                )
        ##### end #####
        return grad, None, None, None, None, None
