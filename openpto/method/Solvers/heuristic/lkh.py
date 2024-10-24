#!/usr/bin/env python
# coding: utf-8
"""

https://github.com/fikisipi/elkai

"""

import elkai
import numpy as np

from openpto.method.Solvers.abcptoSolver import ptoSolver


class LKHSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, n_nodes, **kwargs):
        super().__init__(modelSense)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_vars(self):
        return self.n_nodes**2

    def solve(self, Y):
        return