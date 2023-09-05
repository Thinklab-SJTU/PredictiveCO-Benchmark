from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.method.Solvers.neural.TopKSolver import TopKSolver
from openpto.method.Solvers.neural.BudgetallocSolver import budgetallocSolver
################################# Wrappers ################################################
def solver_wrapper(args, conf, problem):
    return str2solver(args.solver, args.problem, problem)


def str2solver(solver_str, prob_str, problem):
    prob_solver_dict = {
        "budgetalloc":  {"neural": budgetallocSolver},
        # "cubic":CubicTopK,
        # "bipartitematching": BipartiteMatching,
        # 'rmab':RMAB,
        # 'portfolio':PortfolioOpt,
        "cubic": {"neural": TopKSolver},
        "knapsack": {"gurobi": KPGrbSolver},
    }
    # TODO: more problems
    print(prob_str)
    print(solver_str)
    return prob_solver_dict[prob_str][solver_str](**problem.init_API())
