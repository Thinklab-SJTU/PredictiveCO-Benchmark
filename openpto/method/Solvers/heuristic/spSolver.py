#!/usr/bin/env python
# coding: utf-8
"""

"""

import heapq
import itertools

from functools import partial

import numpy as np

from openpto.method.Solvers.abcptoSolver import ptoSolver


class spSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, n_vars, size, neighbourhood_fn, **kwargs):
        super().__init__(modelSense)


    def solve(self, matrix, do_debug=False, **kwargs):
        return


