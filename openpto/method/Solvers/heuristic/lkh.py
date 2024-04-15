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
        self.n_nodes = n_nodes
        self.nodes = list(range(n_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]

    @property
    def n_edges(self):
        return len(self.edges)

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
        tour = cities.solve_tsp()
        sol_array, sol_matrix = self.tour2matrix(
            self.n_nodes, tour
        )  # decode sol to matrix
        others = {"tour": tour, "sol_matrix": sol_matrix}
        return sol_array, None, others

    def tour2matrix(self, n_nodes, tour):
        sol_matrix = np.zeros((n_nodes, n_nodes))
        for idx in range(len(tour) - 1):
            u, v = tour[idx], tour[idx + 1]
            sol_matrix[u, v] = 1
        # sol_array = [sol_matrix[i, j] for i in range(n_nodes) for j in range(n_nodes) if i < j]
        sol_array = np.zeros(self.n_edges, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if sol_matrix[i, j] > 1e-2 or sol_matrix[j, i] > 1e-2:
                sol_array[k] = 1
        return sol_array, sol_matrix
