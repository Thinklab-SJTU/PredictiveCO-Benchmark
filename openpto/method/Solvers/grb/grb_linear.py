#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import gurobipy as gp  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


class LinearGrbSolver(optGrbSolver):

    def __init__(self):
        super().__init__()

    def setObj(self, c):
        return

    def solve(self):
        return
