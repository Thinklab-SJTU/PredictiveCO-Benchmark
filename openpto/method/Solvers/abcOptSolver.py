#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model
"""

from abc import abstractmethod
from copy import deepcopy

from gurobipy import GRB


class optSolver:
    """

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self, modelSense=None):
        # super(optSolver, self).__init__()
        # default sense
        self.modelSense = modelSense if modelSense is not None else GRB.MINIMIZE
        self._model, self.z = self._getModel()

    def __repr__(self):
        return "optSolver " + self.__class__.__name__

    @property
    def num_vars(self):
        """
        number of cost to be predicted
        """
        if hasattr(self, "n_vars"):
            return self.n_vars
        else:
            return len(self.z)

    @abstractmethod
    def _getModel(self):
        """
        An abstract method to build a model from a optimization solver

        Returns:
            tuple: optimization model and variables
        """
        raise NotImplementedError

    # @abstractmethod
    def setObj(self, c):
        """
        An abstract method to set objective function

        Args:
            c (ndarray): cost of objective function
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self):
        """
        An abstract method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        raise NotImplementedError

    def copy(self):
        """
        An abstract method to copy model

        Returns:
            optSolver: new copied model
        """
        new_model = deepcopy(self)
        return new_model

    def addConstr(self, coefs, rhs):
        """
        An abstract method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optSolver: new model with the added constraint
        """
        raise NotImplementedError

    def relax(self):
        """
        A unimplemented method to relax MIP model
        """
        raise RuntimeError("Method 'relax' is not implemented.")
