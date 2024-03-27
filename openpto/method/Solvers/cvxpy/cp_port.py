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
        print("alpha: ", alpha)
        self.num_stocks = num_stocks
        self.solver = self._create_cvxpy_problem(alpha)

    @property
    def num_vars(self):
        return self.num_stocks

    def _create_cvxpy_problem(
        self,
        alpha,
    ):
        x_var = cp.Variable(self.num_stocks)
        L_sqrt_para = cp.Parameter((self.num_stocks, self.num_stocks))
        p_para = cp.Parameter(self.num_stocks)
        constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
        objective = cp.Maximize(
            p_para.T @ x_var - alpha * cp.sum_squares(L_sqrt_para @ x_var)
        )
        problem = cp.Problem(objective, constraints)

        return CvxpyLayer(problem, parameters=[p_para, L_sqrt_para], variables=[x_var])

    def solve(self, Y, sqrt_covar):
        return self.solver(Y, sqrt_covar)
