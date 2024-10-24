#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer

from openpto.method.Solvers.abcptoSolver import ptoSolver


class CpKPSolver(ptoSolver):
    """ """

    def __init__(self, weights, capacity, modelSense, **kwargs):
        super().__init__(modelSense)
        self.weights = weights
        self.capacity = capacity

    @property
    def num_vars(self):
        return len(self.weights)


    def solver_test(
        self,
        p_para,
    ):
        return

    def solve(self, Y, isTrain=True):
        return