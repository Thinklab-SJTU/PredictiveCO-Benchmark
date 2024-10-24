#!/usr/bin/env python
# coding: utf-8
"""
"""

import torch

from openpto.method.Solvers.abcptoSolver import ptoSolver


class TopKSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, n_vars, **kwargs):
        super().__init__(modelSense)

    def solve(self, Y, budget):
        return
