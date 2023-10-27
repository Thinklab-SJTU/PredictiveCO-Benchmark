import gurobipy as gp  # pylint: disable=no-name-in-module
import numpy as np
import torch

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class QPGrbSolver(optGrbSolver):
    def __init__(self, weights, capacity, modelSense, n_items, **kwargs):
        super().__init__()
        self.modelSense = modelSense
        n_items = self.n_items
        self.Q = torch.eye(n_items) / hyperparams["tau"]
        # G = torch.cat((torch.from_numpy(weights).float(), torch.diagflat(torch.ones(n_items)),
        # torch.diagflat(torch.ones(n_items)*-1)), 0)
        # h = torch.cat((torch.tensor([capacity],dtype=torch.float),torch.ones(n_items),torch.zeros(n_items)))

        self.G = torch.from_numpy(weights).float()
        self.h = torch.tensor([capacity], dtype=torch.float)
        self.A = torch.Tensor()
        self.b = torch.Tensor()

    """
    Parameters:
    Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
    p:  A (nBatch, nz) or (nz) Tensor.
    G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
    h:  A (nBatch, nineq) or (nineq) Tensor.
    A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
    b:  A (nBatch, neq) or (neq) Tensor.
    """

    def _getModel(self, Q, p, G, h, A, b, test=False):
        vtype = gp.GRB.CONTINUOUS  # gp.GRB.BINARY if test else gp.GRB.CONTINUOUS
        n = Q.shape[1]
        model = gp.Model()
        model.params.OutputFlag = 0
        x = [
            model.addVar(
                vtype=vtype, name="x_%d" % i, lb=-gp.GRB.INFINITY, ub=+gp.GRB.INFINITY
            )
            for i in range(n)
        ]
        model.update()  # integrate new variables

        # minimize
        #     x.T * Q * x + p * x
        obj = gp.QuadExpr()
        rows, cols = Q.nonzero()
        for i, j in zip(rows, cols):
            obj += x[i] * Q[i, j] * x[j]
        for i in range(n):
            obj += p[i] * x[i]
        model.setObjective(obj, gp.GRB.MINIMIZE)

        # subject to
        #     G * x <= h
        inequality_constraints = []
        if G is not None:
            for i in range(G.shape[0]):
                row = np.where(G[i] != 0)[0]
                inequality_constraints.append(
                    model.addConstr(gp.quicksum(G[i, j] * x[j] for j in row) <= h[i])
                )

        # subject to
        #     A * x == b
        equality_constraints = []
        if A is not None:
            for i in range(A.shape[0]):
                row = np.where(A[i] != 0)[0]
                equality_constraints.append(
                    model.addConstr(gp.quicksum(A[i, j] * x[j] for j in row) == b[i])
                )
        # TODO: CHECK IF NEED UPDATE
        model.update()

        return model

    def solve(self, p, test=False):
        model = self._getModel(self.Q, p, self.G, self.h, self.A, self.b)
        model.optimize()

        x_opt = np.array([x[i].x for i in range(len(x))])
        -(G @ x_opt - h)
        np.array(
            [inequality_constraints[i].pi for i in range(len(inequality_constraints))]
        )
        np.array([equality_constraints[i].pi for i in range(len(equality_constraints))])

        self.obj = model.ObjVal
        self.x = x_opt
        return self.x, self.obj

    # def setObj(self):
    #     """
    #     A method to set objective function

    #     Args:
    #         c (np.ndarray / list): cost of objective function
    #     """
    #     # minimize
    #     #     x.T * Q * x + p * x
    #     obj = gp.QuadExpr()
    #     rows, cols = Q.nonzero()
    #     for i, j in zip(rows, cols):
    #         obj += x[i] * Q[i, j] * x[j]
    #     for i in range(n):
    #         obj += p[i] * x[i]
    #     self._model.setObjective(obj, gp.GRB.MINIMIZE)

    # def solve(self):
    #     obj = gp.QuadExpr()
    #     obj += quadobj
    #     for i in range(len(p)):
    #         obj += p[i] * x[i]
    #     model.setObjective(obj, gp.GRB.MINIMIZE)
    #     model.optimize()
    #     x_opt = np.array([x[i].x for i in range(len(x))])
    #     if G is not None:
    #         slacks = -(G @ x_opt - h)
    #     else:
    #         slacks = np.array([])
    #     lam = np.array(
    #         [inequality_constraints[i].pi for i in range(len(inequality_constraints))]
    #     )
    #     nu = np.array(
    #         [equality_constraints[i].pi for i in range(len(equality_constraints))]
    #     )

    #     others = {"nu": nu, "lam": lam, "slacks": slacks}
    #     return x_opt, model.ObjVal, others
