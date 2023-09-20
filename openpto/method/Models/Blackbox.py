#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Black-box optimization function
"""

import numpy as np
import torch

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Solvers.utils_solver import _solve_in_pass


class blackboxOpt(optModel):
    """
    Reference: <https://arxiv.org/abs/1912.02175>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
        """
        super().__init__(optSolver, processes, solve_ratio)
        # smoothing parameter
        if kwargs["lambd"] <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = kwargs["lambd"]
        # build blackbox optimizer
        self.dbb = blackboxOptFunc()

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
        loss = self.dbb.apply(
            coeff_hat,
            problem,
            params,
            self.optSolver,
            self.processes,
            self.pool,
            self.solve_ratio,
            self.lambd,
            self,
        )
        return loss


class blackboxOptFunc(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        problem,
        params,
        optSolver,
        processes,
        pool,
        solve_ratio,
        lambd,
        module,
    ):
        """
        Forward pass for DBB

        Args:
            coeff_hat (torch.tensor): a batch of predicted values of the cost//coeff_hat
            lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): blackboxOpt module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        cp = coeff_hat.detach().to("cpu").numpy()
        # TODO: remove solution cache
        sols, _ = _solve_in_pass(cp, params, problem, optSolver, processes, pool)
        sol = sols[0]
        # solve
        rand_sigma = np.random.uniform()
        # if rand_sigma <= solve_ratio:
        # sol, _ = _solve_in_pass(cp, params, problem, optSolver, processes, pool)
        # if solve_ratio < 1:
        #     # add into solpool
        #     module.solpool = np.concatenate((module.solpool, sol))
        #     # remove duplicate
        #     module.solpool = np.unique(module.solpool, axis=0)
        # else:
        # sol, _ = _cache_in_pass(cp, optSolver, module.solpool)
        # convert to tensor
        sol = np.array(sol)
        sol_hat = torch.FloatTensor(sol).to(device)
        # save
        ctx.save_for_backward(coeff_hat, sol_hat)
        # add other objects to ctx
        ctx.lambd = lambd
        ctx.optSolver = optSolver
        ctx.processes = processes
        ctx.pool = pool
        ctx.solve_ratio = solve_ratio
        ctx.params = params
        ctx.problem = problem
        ctx.cp = cp
        if solve_ratio < 1:
            ctx.module = module
        ctx.rand_sigma = rand_sigma
        return sol_hat

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DBB
        """
        coeff_hat, sol_hat = ctx.saved_tensors
        lambd = ctx.lambd
        optSolver = ctx.optSolver
        processes = ctx.processes
        params = ctx.params
        problem = ctx.problem
        pool = ctx.pool
        solve_ratio = ctx.solve_ratio
        if solve_ratio < 1:
            pass
        # get device
        device = coeff_hat.device
        # convert tenstor
        cp = coeff_hat.detach().to("cpu").numpy()
        wp = sol_hat.detach().to("cpu").numpy()
        dl = grad_output.detach().to("cpu").numpy()

        ##### work around #####
        # if dl.shape != cp.shape:
        #     dl = np.expand(dl, (*dl.shape, cp.shape[-1]))
        ##### end #####

        # perturbed costs
        ##### work around #####
        if dl.shape != cp.shape:
            cq = cp + lambd * dl.mean(-1, keepdims=True)
        ##### end #####
        else:
            cq = cp + lambd * dl
        # TODO: support batch=1 solution for now
        sols, _ = _solve_in_pass(cq, params, problem, optSolver, processes, pool)
        # solve
        # if rand_sigma <= solve_ratio:
        # sols, _ = _solve_in_pass(cq, params, problem, optSolver, processes, pool)
        # sol = sols[0]
        # if solve_ratio < 1:
        # # add into solpool
        # module.solpool = np.concatenate((module.solpool, sol))
        # # remove duplicate
        # module.solpool = np.unique(module.solpool, axis=0)
        # else:
        #   sol, _ = _cache_in_pass(cq, optSolver, module.solpool)
        # get gradient
        # grad = []
        # for i in range(len(sols)):
        #     grad.append((sols[[i]] - wp[[i]]) / lambd)
        grad = (sols - wp) / lambd
        # convert to tensor
        grad = np.array(grad)
        grad = torch.FloatTensor(grad).to(device)
        ##### work around #####
        cp = ctx.cp
        if grad.shape != cp.shape:
            if np.prod(grad.shape) == np.prod(cp.shape):
                grad = grad.reshape(cp.shape)
            else:
                grad_shape = grad.shape
                grad = grad.view(*grad_shape, 1).expand(*grad_shape, cp.shape[-1])
        ##### end #####
        return grad, None, None, None, None, None, None, None, None
