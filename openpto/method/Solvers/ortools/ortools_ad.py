import torch

from ortools.linear_solver import pywraplp

from openpto.method.Solvers.abcOptSolver import optSolver


# optimization model
class AdOrToolSolver(optSolver):
    def __init__(self, modelSense, **kwargs):
        super().__init__(modelSense)
        self.n_vars = 1

    def solve(self, profits, cost_pv, given_pv):
        # ceil rounded solution
        profits = profits.reshape(-1, 4)
        coefficient = profits.detach().cpu().numpy().tolist()
        num_workers = len(coefficient)
        num_tasks = len(coefficient[0])

        # Solver
        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver("SCIP")

        if not solver:
            return
        # Variables
        # x[i, j] is an array of 0-1 variables, which will be 1
        # if worker i is assigned to task j.
        x = {}
        for worker in range(num_workers):
            for task in range(num_tasks):
                # x[worker, task] = solver.BoolVar(f'x[{worker},{task}]')
                x[worker, task] = solver.NumVar(0, 1.01, f"x[{worker},{task}]")

        # print('Number of variables =', solver.NumVariables())

        # # Each task is assigned to exactly one worker.
        # for task in range(num_tasks):
        #     solver.Add(
        #         solver.Sum([x[worker, task] for worker in range(num_workers)]) == 1)

        solver.Add(
            solver.Sum(
                solver.Sum(x[i, j] * cost_pv[j] for j in range(num_tasks))
                for i in range(num_workers)
            )
            <= given_pv
        )
        # solver.Add(solver.Sum(solver.Sum(x[i,j] * costs2[j] for j in range(num_tasks)) for i in range(num_workers)) <= given_money)

        for worker in range(num_workers):
            solver.Add(solver.Sum([x[worker, task] for task in range(num_tasks)]) == 1)
        # # Each worker is assigned to exactly one task.
        # for worker in range(num_workers):
        #     solver.AddExactlyOne(x[worker, task] for task in range(num_tasks))

        # Objective
        objective_terms = []
        for worker in range(num_workers):
            for task in range(num_tasks):
                objective_terms.append(coefficient[worker][task] * x[worker, task])
        solver.Maximize(solver.Sum(objective_terms))

        # Solve
        status = solver.Solve()
        mockchannels = []
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            solver.Objective().Value()
            # print(f'Total cost = {solver_obj}\n')
            for worker in range(num_workers):
                # task_tmp = list()
                for task in range(num_tasks):
                    # print(f"x: {x[worker, task].solution_value()}")
                    # task_tmp.append(x[worker, task].solution_value())
                    if x[worker, task].solution_value() > 0.5 or task == (num_tasks - 1):
                        mockchannels.append(task)
                        # print(f'Worker {worker} assigned to task {task}.' +
                        #         f' coefficient: {coefficient[worker][task]}')
                        break
                # predchannels.append(task_tmp)
        else:
            print("No solution found.")

        # print(f'Problem solved in {(solver.wall_time()/1000):.3f} seconds')
        x_vec = sol2vec(profits, x)
        # np.hstack(mockchannels)
        return x_vec.reshape(1, -1, 4)


def sol2vec(profits, x):
    num_workers = len(profits)
    num_tasks = len(profits[0])
    sol_res = []
    for worker in range(num_workers):
        worder_res = []
        for task in range(num_tasks):
            worder_res.append(x[worker, task].solution_value())
        sol_res.append(worder_res)
    return torch.FloatTensor(sol_res)
