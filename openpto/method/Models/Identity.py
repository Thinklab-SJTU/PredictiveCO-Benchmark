import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_array, to_device, to_tensor


class IdentitySolver(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
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
            self.ptoSolver,
        )
        return sols_hat


class subopt_Identity(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
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
            self.ptoSolver,
        )
        objs_hat = problem.get_objective(coeff_hat, sols_hat, params)
        # reduction
        loss = do_reduction(objs_hat, hyperparams["reduction"])

        if self.ptoSolver.modelSense == GRB.MINIMIZE:
            pass
        elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
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
        ptoSolver,
    ):
        """ """
        # get device
        device = problem.device
        # convert tenstor
        coeff_hat_array = to_array(coeff_hat)
        # solve
        sols_hat, _ = problem.get_decision(
            to_device(coeff_hat, "cpu"), params, ptoSolver, **problem.init_API()
        )
        sols_hat = to_device(to_tensor(sols_hat), device)
        # add other objects to ctx
        ctx.modelSense = ptoSolver.modelSense
        ctx.coeff_hat_array = coeff_hat_array
        return sols_hat

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for NID
        """
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
