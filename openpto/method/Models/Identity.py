import numpy as np
import torch

from gurobipy import GRB

from openpto.method.Solvers.utils_solver import _solve_in_pass

from .abcOptModel import optModel


class negativeIdentity(optModel):
    """
    An autograd module for differentiable optimizer, which yield optimal a
    solution and use negative identity as gradient on the backward pass.

    For negative identity backpropagation, the objective function is linear and
    constraints are known and fixed, but the cost vector need to be predicted
    from contextual data.

    If the interpolation hyperparameter λ aligns with an appropriate step size,
    then the identity update is tantamount to DBB. However, the identity update
    does not require an additional call to the solver during the backward pass
    and tuning an additional hyperparameter λ.

    Reference: <https://arxiv.org/abs/2205.15213>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optSolver, processes, solve_ratio)
        self.nid = negativeIdentityFunc()

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
        loss = self.nid.apply(
            coeff_hat,
            problem,
            params,
            self.optSolver,
            self.processes,
            self.pool,
            self.solve_ratio,
            self,
        )
        return loss


class negativeIdentityFunc(torch.autograd.Function):
    """
    A autograd function for differentiable black-box optimizer
    """

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
        module,
    ):
        """
        Forward pass for NID

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule):  module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        cp = coeff_hat.detach().to("cpu").numpy()
        # solve
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            sol, _ = _solve_in_pass(cp, params, problem, optSolver, processes, pool)
            if solve_ratio < 1:
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sol))
                # remove duplicates
                module.solpool = np.unique(module.solpool, axis=0)
        # else:
        # sol, _ = _cache_in_pass(cp, optSolver, module.solpool)
        # convert to tensor
        pred_sol = torch.FloatTensor(np.array(sol)).to(device)
        # add other objects to ctx
        ctx.optSolver = optSolver
        ctx.params = params
        ctx.problem = problem
        return pred_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for NID
        """
        optSolver = ctx.optSolver
        # get device
        device = grad_output.device
        # identity matrix
        Ident = torch.eye(grad_output.shape[1]).to(device)
        if optSolver.modelSense == GRB.MINIMIZE:
            grad = -Ident
        if optSolver.modelSense == GRB.MAXIMIZE:
            grad = Ident
        return grad_output @ grad, None, None, None, None, None, None, None
