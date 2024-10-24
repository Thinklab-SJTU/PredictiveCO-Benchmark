import gurobipy as gp  # pylint: disable=no-name-in-module

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class AdGrbSolver(optGrbSolver):
    def __init__(self, modelSense, **kwargs):
        super().__init__(modelSense)
        # turn off output
        self._model.Params.outputFlag = 0

    def _getModel(self, weights, capacity):
        return

    def setObj(self, c):
        return

    def solve(self, Y, **kwargs):
        raise NotImplementedError("The gurobi solver is not supported")
