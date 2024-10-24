import gurobipy as gp  # pylint: disable=no-name-in-module

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class KPGrbSolver(optGrbSolver):
    def __init__(self, weights, capacity, modelSense, **kwargs):
        super().__init__(modelSense)
        self._model, self.z = self._getModel(weights, capacity)
        # turn off output
        self._model.Params.outputFlag = 0

    def _getModel(self, weights, capacity):
        return

    def setObj(self, c):
        return

    def solve(self, y, **kwargs):
        return
