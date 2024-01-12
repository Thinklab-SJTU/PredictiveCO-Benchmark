#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Black-box optimization function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import minus, to_device, to_tensor


class blackbox(optModel):
    """ """

    def __init__(self, optSolver, **kwargs):
        """ """
        super().__init__(optSolver)
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
            coeff_hat, problem, params, self.optSolver, self.lambd, hyperparams
        )
        objs_hat = problem.get_objective(coeff_hat, sols_hat, params, **hyperparams)

        # reduction
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(objs_hat)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(objs_hat)
        elif hyperparams["reduction"] == "none":
            loss = objs_hat
        else:
            raise ValueError(f"No reduction {hyperparams['reduction']}.")

        if self.optSolver.modelSense == GRB.MINIMIZE:
            pass
        elif self.optSolver.modelSense == GRB.MAXIMIZE:
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
        optSolver,
        lambd,
        hyperparams,
    ):
        """
        Forward pass for DBB

        Args:
            coeff_hat (torch.tensor): a batch of predicted values of the cost//coeff_hat
            lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
            optSolver (optModel): an  optimization model

        Returns:
            torch.tensor:  solutions on predicted coefficients
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        coeff_hat_array = coeff_hat.detach().cpu().numpy()
        sols_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, optSolver, **problem.init_API()
        )
        # save to ctx (np.ndarray version)
        ctx.coeff_hat_array = coeff_hat_array
        ctx.sols_hat = sols_hat
        ctx.lambd = lambd
        ctx.optSolver = optSolver
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
        optSolver = ctx.optSolver
        params = ctx.params
        problem = ctx.problem
        # get device
        device = problem.device
        # convert tenstor
        dl = grad_output.detach().cpu().numpy()
        ##### work around #####
        # if dl.shape != coeff_hat_array.shape:
        #     dl = np.expand(dl, (*dl.shape, coeff_hat_array.shape[-1]))
        ##### end #####

        # perturbed costs
        ##### work around #####
        if dl.shape != coeff_hat_array.shape:
            cq = coeff_hat_array + lambd * dl.mean(-1, keepdims=True)
        ##### end #####
        else:
            cq = coeff_hat_array + lambd * dl
        # second np call
        sols_lamb, _ = problem.get_decision(
            to_tensor(cq), params, optSolver, **problem.init_API()
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
