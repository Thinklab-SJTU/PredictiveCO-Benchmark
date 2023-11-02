import gurobipy as gp  # pylint: disable=no-name-in-module
import numpy as np

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class ICONGrbSolver(optGrbSolver):
    def __init__(
        self,
        nbMachines,
        nbTasks,
        nbResources,
        MC,
        U,
        D,
        E,
        L,
        P,
        idle,
        up,
        down,
        q,
        reset=True,
        relax=False,
        verbose=False,
        warmstart=False,
        method=-1,
        **h,
    ):
        self.modelSense = GRB.MINIMIZE
        self.nbMachines = nbMachines
        self.nbTasks = nbTasks
        self.n_vars = nbMachines * nbTasks
        self.nbResources = nbResources
        self.MC = MC
        self.U = U
        self.D = D
        self.E = E
        self.L = L
        self.P = P
        self.idle = idle
        self.up = up
        self.down = down
        self.q = q
        self.relax = relax
        self.verbose = verbose
        self.method = method

        self._getModel()

    def _getModel(self):
        Machines = range(self.nbMachines)
        Tasks = range(self.nbTasks)
        Resources = range(self.nbResources)

        MC = self.MC
        U = self.U
        D = self.D
        E = self.E
        L = self.L
        relax = self.relax
        q = self.q
        N = 1440 // q
        # create a model
        M = gp.Model("icon")
        if not self.verbose:
            M.setParam("OutputFlag", 0)
        if relax:
            x = M.addVars(
                Tasks, Machines, range(N), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x"
            )
        else:
            x = M.addVars(
                Tasks, Machines, range(N), vtype=GRB.BINARY, name="x"
            )  # x是0/1变量

        M.addConstrs(x.sum(f, "*", range(E[f])) == 0 for f in Tasks)  # 46c
        M.addConstrs(x.sum(f, "*", range(L[f] - D[f] + 1, N)) == 0 for f in Tasks)  # 46d
        M.addConstrs(
            (
                gp.quicksum(x[(f, m, t)] for t in range(N) for m in Machines) == 1
                for f in Tasks
            )
        )  # 46b

        # capacity requirement
        for r in Resources:
            for m in Machines:
                for t in range(N):
                    M.addConstr(
                        gp.quicksum(
                            gp.quicksum(
                                x[(f, m, t1)] for t1 in range(max(0, t - D[f] + 1), t + 1)
                            )
                            * U[f][r]
                            for f in Tasks
                        )
                        <= MC[m][r]
                    )
        # M = M.presolve()
        M.update()
        self.model = M

        self.x = dict()
        for var in M.getVars():
            name = var.varName
            if name.startswith("x["):
                (f, m, t) = map(int, name[2:-1].split(","))
                self.x[(f, m, t)] = var

    def solve(self, price, timelimit=None):
        Model = self.model
        D = self.D
        P = self.P
        q = self.q
        N = 1440 // q
        x = self.x
        nbMachines = self.nbMachines
        nbTasks = self.nbTasks
        nbResources = self.nbResources
        Machines = range(nbMachines)
        Tasks = range(nbTasks)
        range(nbResources)

        obj_expr = gp.quicksum(
            [
                x[(f, m, t)] * sum(price[t : t + D[f]]) * P[f] * q / 60
                for f in Tasks
                for t in range(N - D[f] + 1)
                for m in Machines
                if (f, m, t) in x
            ]
        )

        Model.setObjective(obj_expr, GRB.MINIMIZE)
        Model.setParam("Method", self.method)
        Model.optimize()

        solver = np.zeros(N)

        if Model.status in [GRB.Status.OPTIMAL]:
            try:
                task_on = np.zeros((nbTasks, nbMachines, N))
                for (f, m, t), var in x.items():
                    try:
                        task_on[f, m, t] = var.X
                    except AttributeError:
                        task_on[f, m, t] = 0.0
                        print("AttributeError: b' Unable to retrieve attribute 'X'")
                        print("__________Something WRONG___________________________")

                for t in range(N):
                    solver[t] = sum(
                        np.sum(task_on[f, :, max(0, t - D[f] + 1) : t + 1]) * P[f]
                        for f in Tasks
                    )

                solver = solver * q / 60
                self.model.reset(0)
                # print(solver)
                return solver
            except NameError:
                print("\n__________Something wrong_______ \n ")
                # make sure cut is removed! (modifies model)
                self.model.reset(0)
                # print(solver)
                return solver

        elif Model.status == GRB.Status.INF_OR_UNBD:
            print("Model is infeasible or unbounded")
        elif Model.status == GRB.Status.INFEASIBLE:
            print("Model is infeasible")
        elif Model.status == GRB.Status.UNBOUNDED:
            print("Model is unbounded")
        else:
            print("Optimization ended with status %d" % Model.status)
        self.model.reset(0)

        return solver
