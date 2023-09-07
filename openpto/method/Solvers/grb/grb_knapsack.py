import gurobipy as gp

from gurobipy import GRB

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class KPGrbSolver(optGrbSolver):
    def __init__(self, weights, capacity, modelSense):
        # super().__init__() # do not use default __init__ func
        self._model, self.z = self._getModel(weights, capacity)
        self.modelSense = modelSense

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
