#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model
"""

from abc import abstractmethod
from copy import deepcopy



class optSolver(object):
    """1

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self, modelSense):
        # default sense
        self.modelSense = modelSense
        self.z = None

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
