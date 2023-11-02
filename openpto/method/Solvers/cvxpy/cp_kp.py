#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer

from openpto.method.Solvers.cvxpy.cpSolver import optCPSolver


class CpKPSolver(optCPSolver):
    """
    This is an abstract class for Gurobi-based optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self,  weights, capacity, modelSense, **kwargs):
        super().__init__(modelSense)
        self.weights = weights
        self.capacity = capacity
        self.solver = self._create_cvxpy_problem()

    @property
    def num_vars(self):
        return len(self.weights)

    def _create_cvxpy_problem(
        self,
    ):
        x_var = cp.Variable(len(self.weights))
        p_para = cp.Parameter(len(self.weights))
        constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) <= 1]
        # TODO: discrete
        objective = cp.Maximize(
            p_para.T @ x_var
        )
        problem = cp.Problem(objective, constraints)
        return CvxpyLayer(problem, parameters=[p_para], variables=[x_var])

    def solve(self, Y):
        return self.solver(Y) 
