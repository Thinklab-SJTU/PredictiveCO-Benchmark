#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""

from copy import copy

from openpto.method.Solvers.abcOptSolver import optSolver


class optGrbSolver(optSolver):
    """
    This is an abstract class for Gurobi-based optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self):
        super().__init__()
        # model sense
        self.modelSense = self._model.modelSense
        # turn off output
        self._model.Params.outputFlag = 0

    def __repr__(self):
        return "optGRBModel " + self.__class__.__name__

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        others = {}
        return [self.z[k].x for k in self.z], self._model.objVal, others

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        # update model
        self._model.update()
        # new model
        new_model._model = self._model.copy()
        # variables for new model
        x = new_model._model.getVars()
        new_model.x = {key: x[i] for i, key in enumerate(self.z)}
        return new_model
