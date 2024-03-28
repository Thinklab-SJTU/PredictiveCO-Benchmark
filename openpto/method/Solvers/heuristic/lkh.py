#!/usr/bin/env python
# coding: utf-8
"""
"""

import torch
import numpy as np

import elkai
from openpto.method.Solvers.abcptoSolver import ptoSolver


class LKHSolver(ptoSolver):
    """ """

    def __init__(self, modelSense,  n_nodes, **kwargs):
        super().__init__(modelSense)
        self.n_nodes = n_nodes
        self.nodes = list(range(n_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
    
    @property
    def n_vars(self):
        return self.n_nodes**2

    def solve(self, Y):
        """ """
        distance_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for idx, (i, j) in enumerate(self.edges):
            distance_matrix[i][j] = Y[idx]
            distance_matrix[j][i] = Y[idx]
        np.fill_diagonal(distance_matrix, np.inf)
        cities = elkai.DistanceMatrix(distance_matrix)
        sol = cities.solve_tsp()
        z = tour2matrix(self.n_nodes, sol) # decode sol to matrix
        return z, None, None

def tour2matrix(n_nodes, tour):
    sol = np.zeros((n_nodes, n_nodes))
    for idx in range(len(tour)-1):
        u, v = tour[idx], tour[idx+1]
        sol[u,v]= 1
    final_sol = [sol[i,j] for i in range(n_nodes) for j in range(n_nodes) if i < j]
    return final_sol
