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
        self.n_vars = n_vars

    def solve(self, Y, budget):
        """ """
        num_items = Y.shape[1]
        _, idxs = torch.topk(Y, budget, dim=1)
        Z = torch.nn.functional.one_hot(idxs, num_items).sum(dim=-2).sum(-2)
        return Z
