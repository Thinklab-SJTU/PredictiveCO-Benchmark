#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


from openpto.method.Solvers.abcptoSolver import ptoSolver
from openpto.method.Solvers.neural.softTopk import SoftTopk


class softTopkSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, n_vars, **kwargs):
        super().__init__(modelSense)
        self.n_vars = n_vars
        self.topk = SoftTopk()

    def solve(self, Y, budget):
        """ """
        gamma = self.topk(-Y, budget)
        Z = gamma[..., 0].squeeze(-1) * Y.shape[-1]
        return Z
