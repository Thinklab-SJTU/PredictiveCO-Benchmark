import gurobipy as gp  # pylint: disable=no-name-in-module

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class AdGrbSolver(optGrbSolver):
    def __init__(self, modelSense, **kwargs):
        super().__init__(modelSense)
        # self._model, self.z = self._getModel(weights, capacity)
        # turn off output
        self._model.Params.outputFlag = 0

    def _getModel(self, weights, capacity):
        num_items = len(weights)
        # ceate a model
        m = gp.Model()
        m.Params.outputFlag = 0
        x = m.addVars(num_items, name="x", vtype=GRB.BINARY)
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(
            gp.quicksum([weights[i] * x[i] for i in range(num_items)]) <= capacity
        )
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        if len(c) != self.num_vars:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[i] * self.z[k] for i, k in enumerate(self.z))
        self._model.setObjective(obj)

    def solve(self, Y, **kwargs):
        raise NotImplementedError("The gurobi solver is not supported")
