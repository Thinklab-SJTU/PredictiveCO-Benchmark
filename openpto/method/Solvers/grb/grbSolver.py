#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""
from copy import copy

import numpy as np

from openpto.method.Solvers.abcptoSolver import ptoSolver


class optGrbSolver(ptoSolver):
    """ """

    def __init__(self, modelSense):
        super().__init__(modelSense)

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
        solution = np.array([self.z[k].x for k in self.z])
        return solution, self._model.objVal, others

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
