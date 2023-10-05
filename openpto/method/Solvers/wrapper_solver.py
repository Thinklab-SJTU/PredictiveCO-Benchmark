from openpto.method.Solvers.cvxpy.cp_bmatching import BmatchingSolver
from openpto.method.Solvers.grb.grb_advertise import AdGrbSolver
from openpto.method.Solvers.grb.grb_energy import ICONGrbSolver
from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.method.Solvers.heuristic.TopKSolver import TopKSolver
from openpto.method.Solvers.neural.BudgetallocSolver import budgetallocSolver
from openpto.method.Solvers.neural.softTopkSolver import softTopkSolver


################################# Wrappers ################################################
def solver_wrapper(args, conf, problem):
    return str2solver(args, conf, args.solver, args.problem, problem)


def str2solver(args, conf, solver_str, prob_str, problem):
    prob_solver_dict = {
        "budgetalloc": {"neural": budgetallocSolver},
        "bipartitematching": {"cvxpy": BmatchingSolver},
        # 'portfolio':PortfolioOpt,
        "cubic": {"heuristic": TopKSolver, "neural": softTopkSolver},
        "energy": {"gurobi": ICONGrbSolver},
        "knapsack": {"gurobi": KPGrbSolver},
        "advertising": {"gurobi": AdGrbSolver},
    }
    # TODO: more problems
    solve_dict = {**problem.init_API(), **conf["solver"][solver_str]}
    return prob_solver_dict[prob_str][solver_str](**solve_dict)
