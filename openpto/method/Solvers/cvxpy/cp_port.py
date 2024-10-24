#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer

from openpto.method.Solvers.abcptoSolver import ptoSolver


class CpPortfolioSolver(ptoSolver):
    """ """

    def __init__(self, num_stocks, modelSense=None, alpha=1, **kwargs):
        super().__init__(modelSense)
        self.num_stocks = num_stocks

    @property
    def num_vars(self):
        return self.num_stocks
    
    def solve(self, Y, sqrt_covar):
        return
