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

    def __init__(self, weights, capacity, modelSense, **kwargs):
        super().__init__(modelSense)
        self.weights = weights
        self.capacity = capacity
        self.Y = weights
        self.solver_train = self._create_cvxpy_problem_train()
        #self.solver_test = self._create_cvxpy_problem_test()

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
        objective = cp.Maximize(
            p_para.T @ x_var
        )
        problem = cp.Problem(objective, constraints)
        return CvxpyLayer(problem, parameters=[p_para], variables=[x_var])
    
    def _create_cvxpy_problem_test(
        self,
        p_para,
    ):
        x_var = cp.Variable(len(self.weights),boolean=True)
        #p_para = self.Y
        constraints = [self.weights @ x_var <= self.capacity]
        # TODO: discrete
        objective = cp.Maximize(
            p_para.T @ x_var
        )
        problem = cp.Problem(objective, constraints)
        problem.solve()
        #print("x_var.value",x_var.value)
        return x_var.value
    
        

    def solve(self, Y, isTrain=True):
        self.Y=Y
        if isTrain: 
            return self.solver_train(Y) 
        else: 
            return self._create_cvxpy_problem_test(Y)
