#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


from openpto.method.Solvers.abcOptSolver import optSolver
from openpto.method.Solvers.neural.RMABSolver import TopK_custom


class softTopkSolver(optSolver):
    """ """

    def __init__(self, modelSense, n_vars, **kwargs):
        super().__init__(modelSense)
        self.n_vars = n_vars

    def solve(self, Y, budget):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        gamma = TopK_custom(budget)(-Y).squeeze()
        Z = gamma[..., 0] * Y.shape[-1]
        return Z
