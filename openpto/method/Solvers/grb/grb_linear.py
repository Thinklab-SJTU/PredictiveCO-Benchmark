#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import gurobipy as gp  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


class LinearGrbSolver(optGrbSolver):
    """
    This is an general class for for Gurobi-based linear optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self):
        super().__init__()

    def setObj(self, c):
        if len(c) != self.num_vars:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[i] * self.z[k] for i, k in enumerate(self.z))
        self._model.setObjective(obj)

    def solve(self):
        self._model.update()
        self._model.optimize()
        others = {}
        return [self.z[k].x for k in self.z], self._model.objVal, others
