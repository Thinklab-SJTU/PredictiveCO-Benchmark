import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import to_device, to_tensor


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
        sols_hat = self.nid.apply(
            coeff_hat,
            problem,
            params,
            self.optSolver,
        )
        objs_hat = problem.get_objective(coeff_hat, sols_hat, params)
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
    ):
        """
        Forward pass for NID

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optSolver (optModel): an  optimization model

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = coeff_hat.device
        # convert tenstor
        coeff_hat_array = coeff_hat.detach().cpu().numpy()
        # solve
        sols_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, optSolver, **problem.init_API()
        )
        sols_hat = to_device(to_tensor(sols_hat), device)
        # add other objects to ctx
        ctx.modelSense = optSolver.modelSense
        ctx.coeff_hat_array = coeff_hat_array
        return sols_hat

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
        coeff_hat_array = ctx.coeff_hat_array
        # print("coeff_hat_array.shape: ", coeff_hat_array.shape, "grad.shape: ", grad.shape)
        if grad.shape != coeff_hat_array.shape:
            if np.prod(grad.shape) == np.prod(coeff_hat_array.shape):
                grad = grad.reshape(coeff_hat_array.shape)
            else:
                grad_shape = grad.shape
                grad = grad.unsqueeze(-1).expand(*grad_shape, coeff_hat_array.shape[-1])
        ##### end #####
        return grad, None, None, None
