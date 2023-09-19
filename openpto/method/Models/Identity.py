import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Solvers.utils_solver import _solve_in_pass


class negativeIdentity(optModel):
    """
    Reference: <https://arxiv.org/abs/2205.15213>
    """

    def __init__(self, optSolver, processes=1, solve_ratio=1, **kwargs):
        """
        Args:
            optSolver (optModel): an  optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
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
        ctx.modelSense = optSolver.modelSense
        ctx.cp = cp
        return pred_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for NID
        """
        # get device
        # identity matrix
        Ident = grad_output
        # Ident = torch.eye(grad_output.shape[1]).to(device)
        # check the negative
        if ctx.modelSense == GRB.MINIMIZE:
            grad = -Ident
        if ctx.modelSense == GRB.MAXIMIZE:
            grad = Ident
        ##### work around #####
        cp = ctx.cp
        # print("cp.shape: ", cp.shape, "grad.shape: ", grad.shape)
        if grad.shape != cp.shape:
            if np.prod(grad.shape) == np.prod(cp.shape):
                grad = grad.reshape(cp.shape)
            else:
                grad_shape = grad.shape
                grad = grad.unsqueeze(-1).expand(*grad_shape, cp.shape[-1])
        ##### end #####
        return grad, None, None, None, None, None, None, None
