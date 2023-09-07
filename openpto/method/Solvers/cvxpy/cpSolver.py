#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


from openpto.method.Solvers.abcOptSolver import optSolver


class optCPSolver(optSolver):
    """
    This is an abstract class for Gurobi-based optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self):
        super().__init__()
        # model sense
        self._model.update()
        self.modelSense = self._model.modelSense
        # turn off output
        self._model.Params.outputFlag = 0
