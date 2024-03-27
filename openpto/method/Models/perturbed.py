#!/usr/bin/env python
# coding: utf-8
"""
Perturbed optimization function
"""

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction


class perturbed(optModel):
    """
    Reference:
    """

    def __init__(
        self,
        ptoSolver,
        n_samples=10,
        sigma=1.0,
        seed=135,
        **hyperparams,
    ):
        """
        Args:
            ptoSolver (optModel): an  optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            seed (int): random state seed

        """
        super().__init__(ptoSolver)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.rnd = np.random.RandomState(seed)
        # build optimizer
        self.ptb = perturbedOptFunc()
        # solution pool
        n_vars = ptoSolver.num_vars
        self.solpool = np.empty((0, n_vars), dtype=np.float)

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
        sols_hat = self.ptb.apply(
            coeff_hat,
            self.ptoSolver,
            problem,
            params,
            self.n_samples,
            self.sigma,
            self.pool,
            self.rnd,
            self,
        )
        objs_hat = problem.get_objective(coeff_hat, sols_hat, params, **hyperparams)
        # reduction
        loss = do_reduction(objs_hat, hyperparams["reduction"])

        if self.ptoSolver.modelSense == GRB.MINIMIZE:
            pass
        elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
            loss = -loss
        return loss


class perturbedOptFunc(torch.autograd.Function):
    """
    A autograd function for perturbed optimizer
    """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        ptoSolver,
        problem,
        params,
        n_samples,
        sigma,
        pool,
        rnd,
        module,
    ):
        """
        Forward pass for perturbed

        Args:
            coeff_hat (torch.tensor): a batch of predicted values of the cost
            ptoSolver (optModel): an  optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            pool (ProcessPool): process pool object
            rnd (RondomState): numpy random state

            module (optModel): perturbedOpt module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        cp = coeff_hat.detach().cpu().numpy()
        # sample perturbations
        noises = rnd.normal(0, 1, size=(n_samples, *cp.shape[1:]))
        coeff_perturb = cp + sigma * noises
        # solve with perturbation
        ptb_sols, _ = problem.get_decision(
            coeff_perturb, params, ptoSolver, **problem.init_API()
        )
        # add into solpool
        module.solpool = np.concatenate((module.solpool, ptb_sols))
        # remove duplicate
        module.solpool = np.unique(module.solpool, axis=0)
        # rand_sigma = np.random.uniform()
        # solution expectation
        e_sol = ptb_sols.mean(axis=1, keepdims=True)
        # convert to tensor
        noises = torch.from_numpy(noises).to(device)
        ptb_sols = torch.from_numpy(ptb_sols).to(device)
        e_sol = torch.from_numpy(e_sol).to(device)
        # save solutions
        ctx.save_for_backward(ptb_sols, noises)
        # add other objects to ctx
        ctx.n_samples = n_samples
        ctx.sigma = sigma
        return e_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed
        """
        ptb_sols, noises = ctx.saved_tensors
        n_samples = ctx.n_samples
        sigma = ctx.sigma
        grad = torch.einsum(
            "nbd,bn->bd", noises, torch.einsum("bnd,bd->bn", ptb_sols, grad_output)
        )
        grad /= n_samples * sigma
        return grad, None, None, None, None, None, None, None, None


# class perturbedFenchelYoung(optModel):
#     """
#     An autograd module for Fenchel-Young loss using perturbation techniques. The
#     use of the loss improves the algorithmic by the specific expression of the
#     gradients of the loss.

#     For the perturbed optimizer, the cost vector need to be predicted from
#     contextual data and are perturbed with Gaussian noise.

#     The Fenchel-Young loss allows to directly optimize a loss between the features
#     and solutions with less computation. Thus, allows us to design an algorithm
#     based on stochastic gradient descent.

#     Reference:
#     """

#     def __init__(
#         self,
#         ptoSolver,
#         n_samples=10,
#         sigma=1.0,
#         seed=135,
#         dataset=None,
#     ):
#         """
#         Args:
#             ptoSolver (optModel): an  optimization model
#             n_samples (int): number of Monte-Carlo samples
#             sigma (float): the amplitude of the perturbation
#             seed (int): random state seed
#
#             dataset (None/optDataset): the training data
#         """
#         super().__init__(ptoSolver)
#         # number of samples
#         self.n_samples = n_samples
#         # perturbation amplitude
#         self.sigma = sigma
#         # random state
#         self.rnd = np.random.RandomState(seed)
#         # build optimizer
#         self.pfy = perturbedFenchelYoungFunc()

#     def forward(self, coeff_hat, true_sol, reduction="mean"):
#         """
#         Forward pass
#         """
#         loss = self.pfy.apply(
#             coeff_hat,
#             true_sol,
#             self.ptoSolver,
#             self.n_samples,
#             self.sigma,
#             self.pool,
#             self.rnd,
#             self,
#         )
#         # reduction
#         loss = do_reduction(loss, hyperparams["reduction"])
#         return loss


# class perturbedFenchelYoungFunc(torch.autograd.Function):
#     """
#     A autograd function for Fenchel-Young loss using perturbation techniques.
#     """

#     @staticmethod
#     def forward(
#         ctx,
#         coeff_hat,
#         true_sol,
#         ptoSolver,
#         n_samples,
#         sigma,
#         pool,
#         rnd,
#         module,
#     ):
#         """
#         Forward pass for perturbed Fenchel-Young loss

#         Args:
#             coeff_hat (torch.tensor): a batch of predicted values of the cost
#             true_sol (torch.tensor): a batch of true optimal solutions
#             ptoSolver (optModel): an  optimization model
#             n_samples (int): number of Monte-Carlo samples
#             sigma (float): the amplitude of the perturbation
#             pool (ProcessPool): process pool object
#             rnd (RondomState): numpy random state
#
#             module (optModel): perturbedFenchelYoung module

#         Returns:
#             torch.tensor: solution expectations with perturbation
#         """
#         # get device
#         device = coeff_hat.device
#         # convert tenstor
#         cp = coeff_hat.detach().cpu().numpy()
#         w = true_sol.detach().cpu().numpy()
#         # sample perturbations
#         noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
#         ptb_c = cp + sigma * noises
#         # solve with perturbation
#         rand_sigma = np.random.uniform()
#         ptb_sols = _solve_in_pass(ptb_c, ptoSolver, pool)
#         sols = ptb_sols.reshape(-1, cp.shape[1])
#         # add into solpool
#         module.solpool = np.concatenate((module.solpool, sols))
#         # remove duplicate
#         module.solpool = np.unique(module.solpool, axis=0)
#         # solution expectation
#         e_sol = ptb_sols.mean(axis=1)
#         # difference
#         if ptoSolver.modelSense == GRB.MINIMIZE:
#             diff = w - e_sol
#         if ptoSolver.modelSense == GRB.MAXIMIZE:
#             diff = e_sol - w
#         # loss
#         loss = np.sum(diff**2, axis=1)
#         # convert to tensor
#         diff = torch.FloatTensor(diff).to(device)
#         loss = torch.FloatTensor(loss).to(device)
#         # save solutions
#         ctx.save_for_backward(diff)
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass for perturbed Fenchel-Young loss
#         """
#         (grad,) = ctx.saved_tensors
#         grad_output = torch.unsqueeze(grad_output, dim=-1)
#         return grad * grad_output, None, None, None, None, None, None, None, None, None


# def _solve_in_pass(ptb_c, ptoSolver, pool):
#     """
#     A function to solve optimization in the forward pass
#     """
#     # number of instance
#     n_samples, ins_num = ptb_c.shape[0], ptb_c.shape[1]
#     # single-core
#     if processes == 1:
#         ptb_sols = []
#         for i in range(ins_num):
#             sols = []
#             # per sample
#             for j in range(n_samples):
#                 # solve
#                 ptoSolver.setObj(ptb_c[j, i])
#                 sol, _ = ptoSolver.solve()
#                 sols.append(sol)
#             ptb_sols.append(sols)
#     # multi-core
#     else:
#         # get class
#         model_type = type(ptoSolver)
#         # get args
#         args = getArgs(ptoSolver)
#         # parallel computing
#         ptb_sols = pool.amap(
#             _solveWithObj4Par,
#             ptb_c.transpose(1, 0, 2),
#             [args] * ins_num,
#             [model_type] * ins_num,
#         ).get()
#     return np.array(ptb_sols)


# def _cache_in_pass(ptb_c, ptoSolver, solpool):
#     """
#     A function to use solution pool in the forward/backward pass
#     """
#     # number of samples & instance
#     n_samples, ins_num, _ = ptb_c.shape
#     # init sols
#     ptb_sols = []
#     for j in range(n_samples):
#         # best solution in pool
#         solpool_obj = ptb_c[j] @ solpool.T
#         if ptoSolver.modelSense == GRB.MINIMIZE:
#             ind = np.argmin(solpool_obj, axis=1)
#         if ptoSolver.modelSense == GRB.MAXIMIZE:
#             ind = np.argmax(solpool_obj, axis=1)
#         ptb_sols.append(solpool[ind])
#     return np.array(ptb_sols).transpose(1, 0, 2)


# def _solveWithObj4Par(perturbed_costs, args, model_type):
#     """
#     A global function to solve function in parallel processors

#     Args:
#         perturbed_costs (np.ndarray): costsof objective function with perturbation
#         args (dict): optModel args
#         model_type (ABCMeta): optModel class type

#     Returns:
#         list: optimal solution
#     """
#     # rebuild model
#     ptoSolver = model_type(**args)
#     # per sample
#     sols = []
#     for cost in perturbed_costs:
#         # set obj
#         ptoSolver.setObj(cost)
#         # solve
#         sol, _ = ptoSolver.solve()
#         sols.append(sol)
#     return sols
