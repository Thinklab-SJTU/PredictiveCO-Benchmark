#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model
"""

from abc import abstractmethod


class ptoSolver(object):
    """ """

    def __init__(self, modelSense):
        # default sense
        self.modelSense = modelSense
        self.z = None

    def __repr__(self):
        return "ptoSolver " + self.__class__.__name__

    @property
    def num_vars(self):
        """
        number of cost to be predicted
        """
        if hasattr(self, "n_vars"):
            return self.n_vars
        else:
            return len(self.z)

    @abstractmethod
    def solve(self):
        """
        An abstract method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        raise NotImplementedError
