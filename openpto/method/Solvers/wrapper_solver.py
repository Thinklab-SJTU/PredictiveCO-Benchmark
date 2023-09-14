from openpto.method.Solvers.grb.grb_energy import ICONGrbSolver
from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.method.Solvers.neural.BudgetallocSolver import budgetallocSolver
from openpto.method.Solvers.neural.TopKSolver import TopKSolver
from openpto.method.Solvers.cvxpy.cp_bmatching import BmatchingSolver


################################# Wrappers ################################################
def solver_wrapper(args, conf, problem):
    return str2solver(args.solver, args.problem, problem)


def str2solver(solver_str, prob_str, problem):
    prob_solver_dict = {
        "budgetalloc": {"neural": budgetallocSolver},
        "bipartitematching":{"cvxpy":BmatchingSolver},
        # "bipartitematching": BipartiteMatching,
        # 'rmab':RMAB,
        # 'portfolio':PortfolioOpt,
        "cubic": {"neural": TopKSolver},
        "energy": {"gurobi": ICONGrbSolver},
        "knapsack": {"gurobi": KPGrbSolver},
    }
    # TODO: more problems
    return prob_solver_dict[prob_str][solver_str](**problem.init_API())
