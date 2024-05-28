import torch

from ortools.linear_solver import pywraplp  # pylint: disable=no-name-in-module

from openpto.method.Solvers.abcptoSolver import ptoSolver
from openpto.method.utils_method import to_array


# optimization model
class AdOrToolSolver(ptoSolver):
    def __init__(self, modelSense, n_combs, **kwargs):
        super().__init__(modelSense)
        self.n_vars = 1  # TODO: this is not used
        self.n_combs = n_combs

    def solve(self, profits, cost_pv, given_pv):
        # TODO: only support batch size = 1
        # ceil rounded solution
        profits = profits.reshape(-1, self.n_combs)
        coefficient = to_array(profits)
        num_users = len(coefficient)
        num_channels = len(coefficient[0])

        # Solver
        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver("SCIP")

        if not solver:
            return
        # Variables
        # x[i, j] is an array of 0-1 variables, which will be 1
        # if worker i is assigned to task j.
        x = {}
        for worker in range(num_users):
            for task in range(num_channels):
                x[worker, task] = solver.BoolVar(f"x[{worker},{task}]")
                # x[worker, task] = solver.NumVar(0, 1.01, f"x[{worker},{task}]")

        # print('Number of variables =', solver.NumVariables())

        # # Each task is assigned to exactly one worker.
        # for task in range(num_channels):
        #     solver.Add(
        #         solver.Sum([x[worker, task] for worker in range(num_users)]) == 1)

        solver.Add(
            solver.Sum(
                solver.Sum(x[i, j] * cost_pv[j] for j in range(num_channels))
                for i in range(num_users)
            )
            <= given_pv
        )
        # solver.Add(solver.Sum(solver.Sum(x[i,j] * costs2[j] for j in range(num_channels)) for i in range(num_users)) <= given_money)

        for worker in range(num_users):
            solver.Add(solver.Sum([x[worker, task] for task in range(num_channels)]) == 1)
        # # Each worker is assigned to exactly one task.
        # for worker in range(num_users):
        #     solver.AddExactlyOne(x[worker, task] for task in range(num_channels))

        # Objective
        objective_terms = []
        for worker in range(num_users):
            for task in range(num_channels):
                objective_terms.append(coefficient[worker][task] * x[worker, task])
        solver.Maximize(solver.Sum(objective_terms))

        # Solve
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # solver.Objective().Value()
            pass
        else:
            raise ValueError("No solution found.")

        # print(f'Problem solved in {(solver.wall_time()/1000):.3f} seconds')
        return sol2vec(x, num_users, num_channels)


def sol2vec(x, num_users, num_channels):
    sol_res = []
    for worker in range(num_users):
        for task in range(num_channels):
            sol_res.append(x[worker, task].solution_value())
    # TODO: only support batch=1
    return torch.FloatTensor(sol_res).reshape(-1)  # num_users * num_channels)
