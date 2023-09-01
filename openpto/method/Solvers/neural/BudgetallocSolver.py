#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""

from copy import copy
import torch
import gurobipy as gp
from openpto.method.Solvers.abcOptSolver import optSolver
from openpto.method.Solvers.neural.SubmodularOptimizer import SubmodularOptimizer


class budgetallocSolver(optSolver):
    """
    """
    def _getModel(self):
        return None, None
    
    def get_objective(self, Y, Z, w=None, **kwargs):
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        assert Y.shape[-2] == Z.shape[-1]
        assert len(Z.shape) + 1 == len(Y.shape)

        # Initialise weights to default value
        if w is None:
            w = torch.ones(Y.shape[-1]).requires_grad_(False).to(Z.device)
        else:
            assert Y.shape[-1] == w.shape[0]
            assert len(w.shape) == 1

        # Calculate objective
        p_fail = 1 - Z.unsqueeze(-1) * Y
        p_all_fail = p_fail.prod(dim=-2)
        obj = (w * (1 - p_all_fail)).sum(dim=-1)
        return obj
    
    def solve(self, Y, Z_init=None, **kwargs): 
        self.budget = 2
        self.opt = SubmodularOptimizer(self.get_objective, self.budget)
        if len(Y.shape) == 2:
            Z=self.opt(Y, Z_init=Z_init)
            obj=self.get_objective(Y,Z)
            #print("Z_shape=",Z.shape,"obj_shape=",obj.shape)
            #print(Z)
            #print(obj)
            return Z,obj
        # If it's not...
        #   Remember the shape
        Y_shape = Y.shape
        #   Break it down into individual instances and solve
        Y_new = Y.view((-1, Y_shape[-2], Y_shape[-1]))
        Z = torch.cat([self.opt(y, Z_init=Z_init) for y in Y_new], dim=0)
        #   Convert it back to the right shape
        Z = Z.view((*Y_shape[:-2], -1))
        print(Z.shape)
        return Z
    
    def setObj(self, Y):
        return None

  
