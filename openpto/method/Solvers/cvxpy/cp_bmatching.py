#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer

from openpto.method.Solvers.abcptoSolver import ptoSolver


class BmatchingSolver(ptoSolver):
    """ """

    def __init__(self, modelSense=None, isTrain=True, num_nodes=50, **kwargs):
        super().__init__(modelSense)
        self.num_nodes = num_nodes

    @property
    def num_vars(self):
        return self.num_nodes * self.num_nodes

    def _getModel(self, isTrain=True, num_nodes=50):
        return


    def solve(self, **kwargs):
        raise NotImplementedError
