#!/usr/bin/env python
# coding: utf-8

import torch

from openpto.method.Solvers.abcptoSolver import ptoSolver
from openpto.method.Solvers.neural.submodular import OptimiseSubmodular


class budgetallocSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, n_vars, get_objective, num_iters, budget, **kwargs):
        super().__init__(modelSense)
        self.n_vars = n_vars
        self.budget = budget
        self.opt = SubmodularOptimizer(get_obj=get_objective, num_iters=num_iters)

    def solve(self, y, Z_init=None):
        return self.opt(y, self.budget, Z_init=Z_init)


class SubmodularOptimizer(torch.nn.Module):
    """
    Wrapper around OptimiseSubmodular that saves state information.
    """

    def __init__(
        self,
        get_obj,  # A function that returns the value of the objective we want to minimise
        lr=0.1,  # learning rate for optimiser
        momentum=0.9,  # momentum for optimiser
        num_iters=100,  # number of optimisation steps
        verbose=False,  # print intermediate solution statistics
    ):
        super(SubmodularOptimizer, self).__init__()
        self.get_obj = get_obj
        self.lr = lr
        self.momentum = momentum
        self.num_iters = num_iters
        self.verbose = verbose

    def forward(
        self,
        Yhat,
        budget,
        Z_init=None,  # value with which to warm start Z
    ):
        """
        Computes the optimal Z for the predicted Yhat using the supplied optimizer.
        """
        Z = OptimiseSubmodular.apply(
            Yhat,
            self.get_obj,
            budget,
            self.lr,
            self.momentum,
            self.num_iters,
            self.verbose,
            Z_init,
        )
        return Z


# Unit test for submodular optimiser
if __name__ == "__main__":
    # Unit Test
    def get_obj(Y, Z):
        # Function to be *maximised*
        #   Sanity check inputs
        assert Y.shape[0] == Z.shape[0]
        assert Z.ndim == 1

        #   Compute submodular objective from Wilder, et. al. (2019)
        p_fail = 1 - Z.unsqueeze(1) * Y
        p_all_fail = p_fail.prod(dim=0)
        obj = (1 - p_all_fail).sum()
        return obj

    #   Load class
    opt = SubmodularOptimizer(get_obj, budget=1)

    #   Genereate data
    torch.manual_seed(100)
    Y = torch.rand((5, 10), requires_grad=True)

    #   Perform forward pass
    Z = opt(Y)
    loss = get_obj(Y, Z)
    print(loss)

    # Perform backward pass
    loss.backward()

    # Use torch.gradcheck to double check gradients
    torch.autograd.gradcheck(opt, Y)
