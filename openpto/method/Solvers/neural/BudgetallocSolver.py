#!/usr/bin/env python
# coding: utf-8

import torch

from openpto.method.Solvers.abcptoSolver import ptoSolver
from openpto.method.Solvers.neural.submodular import OptimiseSubmodular


class budgetallocSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, n_vars, get_objective, num_iters, budget, **kwargs):
        super().__init__(modelSense)

    def solve(self, y, Z_init=None):
        return


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


    def forward(
        self,
        Yhat,
        budget,
        Z_init=None,  # value with which to warm start Z
    ):
        return

