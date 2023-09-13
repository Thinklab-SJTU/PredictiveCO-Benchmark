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

    def __init__(self, modelSense):
        super().__init__(modelSense)
