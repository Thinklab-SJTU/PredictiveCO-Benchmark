from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver


################################# Wrappers ################################################
def solver_wrapper(args, conf, problem):
    return str2solver(args.solver, args.problem, problem)


def str2solver(solver_str, prob_str, problem):
    prob_solver_dict = {
        # "budgetalloc": BudgetAllocation,
        # "cubic":CubicTopK,
        # "bipartitematching": BipartiteMatching,
        # 'rmab':RMAB,
        # 'portfolio':PortfolioOpt,
        "knapsack": {"gurobi": KPGrbSolver},
    }
    # TODO: more problems
    return prob_solver_dict[prob_str][solver_str](**problem.init_API())
