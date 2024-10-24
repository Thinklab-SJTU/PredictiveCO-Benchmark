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
        return

    def solve(self):
        return

    def copy(self):
        return
