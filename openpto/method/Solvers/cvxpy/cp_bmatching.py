#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer
from gurobipy import GRB 
from openpto.method.Solvers.cvxpy.cpSolver import optCPSolver


class BmatchingSolver(optCPSolver):
    """
    This is an abstract class for Gurobi-based optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self, modelSense=None, isTrain=True, num_nodes=50, **kwargs):
        super().__init__(modelSense)
        self.num_nodes=num_nodes
    
    @property
    def num_vars(self):
        return self.num_nodes*self.num_nodes

    def _getModel(self, isTrain=True, num_nodes=50):
        return self._create_cvxpy_problem(isTrain, num_nodes)

    def _create_cvxpy_problem(
        self,
        isTrain=True,
        num_nodes=50,
        gamma=0.1,
    ):
        # Variables
        Z = cp.Variable((num_nodes, num_nodes), nonneg=True)
        Y = cp.Parameter((num_nodes, num_nodes))

        # Objective
        matching_obj = cp.sum(cp.multiply(Z, Y))
        reg = cp.norm(Z) if isTrain else 0
        objective = cp.Maximize(matching_obj - gamma * reg)

        # Flow Constraints
        constraints = [cp.sum(Z, axis=0) == 1, cp.sum(Z, axis=1) == 1]

        # Problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        return CvxpyLayer(problem, parameters=[Y], variables=[Z])
