#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


from openpto.method.Solvers.cvxpy.cpSolver import optCPSolver
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

class BmatchingSolver(optCPSolver):
    """
    This is an abstract class for Gurobi-based optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self, modelSense=None):
        super().__init__()
        self.modelSense=modelSense
        self._getModel()
    
    def _getModel(self):
        self._create_cvxpy_problem()
        
    def _create_cvxpy_problem(
        self,
        isTrain=True,
        gamma=0.1,
    ):
        # Variables
        Z = cp.Variable((self.num_nodes, self.num_nodes), nonneg=True)
        Y = cp.Parameter((self.num_nodes, self.num_nodes))

        # Objective
        matching_obj = cp.sum( cp.multiply(Z, Y) )
        reg = cp.norm(Z) if isTrain else 0
        objective = cp.Maximize( matching_obj - gamma * reg )

        # Flow Constraints
        constraints = [cp.sum(Z, axis=0) == 1, cp.sum(Z, axis=1) == 1]

        # Problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        return CvxpyLayer(problem, parameters=[Y], variables=[Z])
        
        
        
    
    