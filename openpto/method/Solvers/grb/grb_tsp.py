from collections import defaultdict

import gurobipy as gp  # pylint: disable=no-name-in-module
import numpy as np

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class TSPGrbSolver(optGrbSolver):
    """
    Code is adopted by PyEPO. This class is optimization model for traveling salesman problem based on Gavish–Graves (GG) formulation.
    """

    def __init__(self, modelSense, n_nodes, **kwargs):
        super().__init__(modelSense)
        self.n_nodes = n_nodes
        # TSP nodes & edges
        self.nodes = list(range(n_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        self._model, self.z = self._getModel()
        # turn off output
        self._model.Params.outputFlag = 0

    @property
    def n_edges(self):
        return len(self.edges)

    def _getModel(
        self,
    ):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("tsp")
        # varibles
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        z = m.addVars(directed_edges, name="z", vtype=GRB.BINARY)
        y = m.addVars(directed_edges, name="y")
        # sense
        assert self.modelSense == GRB.MINIMIZE
        m.modelSense = self.modelSense
        # constraints
        m.addConstrs(z.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(z.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(
            y.sum(i, "*") - gp.quicksum(y[j, i] for j in self.nodes[1:] if j != i) == 1
            for i in self.nodes[1:]
        )
        m.addConstrs(y[i, j] <= (len(self.nodes) - 1) * z[i, j] for (i, j) in z if i != 0)
        return m, z

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.n_edges:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(
            c[k] * (self.z[i, j] + self.z[j, i]) for k, (i, j) in enumerate(self.edges)
        )
        self._model.setObjective(obj)
        return

    def solve(self, y, **kwargs):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self.setObj(y)
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.n_edges, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if self.z[i, j].x > 1e-2 or self.z[j, i].x > 1e-2:
                sol[k] = 1
        others = {}
        return sol, self._model.objVal, others

    def grbarr2arr(self, grb_tensor):
        sol_array = np.zeros_like(grb_tensor)
        for k, (i, j) in enumerate(self.edges):
            sol_array[i, j] = self.z[i, j].x
            sol_array[j, i] = self.z[j, i].x
        return sol_array


def getTour(sol, edges):
    """
    A method to get a tour from solution

    Args:
        sol (list): solution

    Returns:
        list: a TSP tour
    """
    # active edges
    edges = defaultdict(list)
    for i, (j, k) in enumerate(edges):
        if sol[i] > 1e-2:
            edges[j].append(k)
            edges[k].append(j)
    # get tour
    visited = {list(edges.keys())[0]}
    tour = [list(edges.keys())[0]]
    while len(visited) < len(edges):
        i = tour[-1]
        for j in edges[i]:
            if j not in visited:
                tour.append(j)
                visited.add(j)
                break
    if 0 in edges[tour[-1]]:
        tour.append(0)
    return tour
