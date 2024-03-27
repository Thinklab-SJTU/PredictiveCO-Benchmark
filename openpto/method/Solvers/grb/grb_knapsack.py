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
        num_items = len(weights)
        m = gp.Model()
        x = m.addVars(num_items, name="x", vtype=GRB.BINARY)
        m.modelSense = self.modelSense
        m.addConstr(
            gp.quicksum([weights[i] * x[i] for i in range(num_items)]) <= capacity
        )
        return m, x

    def setObj(self, c):
        if len(c) != self.num_vars:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[i] * self.z[k] for i, k in enumerate(self.z))
        self._model.setObjective(obj)

    def solve(self, y, **kwargs):
        self.setObj(y)
        self._model.update()
        self._model.optimize()
        others = {}
        return [self.z[k].x for k in self.z], self._model.objVal, others
