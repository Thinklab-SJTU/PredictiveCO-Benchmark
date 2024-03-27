#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer

from openpto.method.Solvers.abcptoSolver import ptoSolver


class CpKPSolver(ptoSolver):
    """ """

    def __init__(self, weights, capacity, modelSense, **kwargs):
        super().__init__(modelSense)
        self.weights = weights
        self.capacity = capacity
        self.solver_train = self._create_cvxpy_problem_train()

    @property
    def num_vars(self):
        return len(self.weights)

    def _create_cvxpy_problem_train(
        self,
    ):
        x_var = cp.Variable(len(self.weights))
        p_para = cp.Parameter(len(self.weights))
        constraints = [x_var >= 0, x_var <= 1, self.weights @ x_var <= self.capacity]
        # TODO: discrete
        objective = cp.Maximize(p_para.T @ x_var)
        problem = cp.Problem(objective, constraints)
        return CvxpyLayer(problem, parameters=[p_para], variables=[x_var])

    def solver_test(
        self,
        p_para,
    ):
        self.weights = self.weights.cpu()
        x_var = cp.Variable(len(self.weights), boolean=True)
        constraints = [self.weights @ x_var <= self.capacity]
        objective = cp.Maximize(p_para.T @ x_var)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        # print("x_var.value",x_var.value)
        return x_var.value

    def solve(self, Y, isTrain=True):
        Y = Y.squeeze(-1)
        if isTrain:
            return self.solver_train(Y)
        else:
            return self.solver_test(Y)
