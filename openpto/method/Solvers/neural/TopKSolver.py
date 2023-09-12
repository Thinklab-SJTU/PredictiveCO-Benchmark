#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""

import torch

from openpto.method.Solvers.abcOptSolver import optSolver


class TopKSolver(optSolver):
    """ """

    def __init__(self, modelSense, n_vars):
        super().__init__(modelSense)
        self.n_vars = n_vars

    def _getModel(self):
        return None, None

    def solve(self, Y):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        self.budget = 1
        Y = torch.from_numpy(Y)
        _, indices = torch.topk(Y, k=self.budget, dim=-1)
        top_k_one_hot = torch.zeros_like(Y)
        top_k_one_hot.scatter_(-1, indices, 1)
        return top_k_one_hot, (top_k_one_hot * Y).sum(dim=-1)

    def setObj(self, Y):
        return None
