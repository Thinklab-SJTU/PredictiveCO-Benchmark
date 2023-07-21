import numpy as np

import gurobipy as gp
from gurobipy import GRB

from openpto.method.Solver.grb.grbSolver import optGrbSolver


# optimization model
class KPGrbSolver(optGrbSolver):
    def __init__(self, weights, capacity):
        # super().__init__()
        self._model, self.x = self._getModel(weights, capacity)

    def _getModel(self, weights, capacity):
        num_items = len(weights)
        # ceate a model
        m = gp.Model()
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(num_items, name="x", vtype=GRB.BINARY)
        # sense (must be minimize)
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([weights[i] * x[i] for i in range(num_items)]) <= capacity)
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.x))
        self._model.setObjective(obj)