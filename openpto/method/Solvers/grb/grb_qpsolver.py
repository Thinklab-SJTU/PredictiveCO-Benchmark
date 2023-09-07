# import gurobipy as gp
# import numpy as np

# from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# # optimization model
# class QPGrbSolver(optGrbSolver):
#     def __init__(self, modelSense):
#         super().__init__()
#         self._model, self.x = self._getModel()
#         self.modelSense = modelSense

#     def _getModel(
#         self,
#     ):
#         """
#         Convert to Gurobi model. Copied from
#         https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/gurobi_.py
#         """
#         vtype = gp.GRB.CONTINUOUS  # gp.GRB.BINARY if test else gp.GRB.CONTINUOUS
#         n = Q.shape[1]
#         model = gp.Model()
#         model.params.OutputFlag = 0
#         x = [
#             model.addVar(
#                 vtype=vtype, name="x_%d" % i, lb=-gp.GRB.INFINITY, ub=+gp.GRB.INFINITY
#             )
#             for i in range(n)
#         ]
#         # model.update()   # integrate new variables

#         # subject to
#         #     G * x <= h
#         inequality_constraints = []
#         if G is not None:
#             for i in range(G.shape[0]):
#                 row = np.where(G[i] != 0)[0]
#                 inequality_constraints.append(
#                     model.addConstr(gp.quicksum(G[i, j] * x[j] for j in row) <= h[i])
#                 )

#         # subject to
#         #     A * x == b
#         equality_constraints = []
#         if A is not None:
#             for i in range(A.shape[0]):
#                 row = np.where(A[i] != 0)[0]
#                 equality_constraints.append(
#                     model.addConstr(gp.quicksum(A[i, j] * x[j] for j in row) == b[i])
#                 )

#         model.optimize()

#         x_opt = np.array([x[i].x for i in range(len(x))])
#         slacks = -(G @ x_opt - h)
#         lam = np.array(
#             [inequality_constraints[i].pi for i in range(len(inequality_constraints))]
#         )
#         nu = np.array(
#             [equality_constraints[i].pi for i in range(len(equality_constraints))]
#         )

#         return model.ObjVal, x_opt, nu, lam, slacks
#         return model, x

#     def setObj(self, c):
#         """
#         A method to set objective function

#         Args:
#             c (np.ndarray / list): cost of objective function
#         """
#         # minimize
#         #     x.T * Q * x + p * x
#         obj = gp.QuadExpr()
#         rows, cols = Q.nonzero()
#         for i, j in zip(rows, cols):
#             obj += x[i] * Q[i, j] * x[j]
#         for i in range(n):
#             obj += p[i] * x[i]
#         self._model.setObjective(obj, gp.GRB.MINIMIZE)

#     def solve(self):
#         obj = gp.QuadExpr()
#         obj += quadobj
#         for i in range(len(p)):
#             obj += p[i] * x[i]
#         model.setObjective(obj, gp.GRB.MINIMIZE)
#         model.optimize()
#         x_opt = np.array([x[i].x for i in range(len(x))])
#         if G is not None:
#             slacks = -(G @ x_opt - h)
#         else:
#             slacks = np.array([])
#         lam = np.array(
#             [inequality_constraints[i].pi for i in range(len(inequality_constraints))]
#         )
#         nu = np.array(
#             [equality_constraints[i].pi for i in range(len(equality_constraints))]
#         )

#         others = {"nu": nu, "lam": lam, "slacks": slacks}
#         return x_opt, model.ObjVal, others
