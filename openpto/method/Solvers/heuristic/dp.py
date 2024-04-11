#!/usr/bin/env python
# coding: utf-8
"""
"""


from openpto.method.Solvers.abcptoSolver import ptoSolver


class DPSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, **kwargs):
        super().__init__(modelSense)

    def solve(self, Y):
        """ """
        z = 0
        raise NotImplementedError
        return z, None, None
