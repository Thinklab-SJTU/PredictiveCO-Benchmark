from openpto.method.Solvers.cvxpy.cp_bmatching import BmatchingSolver
from openpto.method.Solvers.cvxpy.cp_kp import CpKPSolver
from openpto.method.Solvers.cvxpy.cp_port import CpPortfolioSolver
from openpto.method.Solvers.grb.grb_advertise import AdGrbSolver
from openpto.method.Solvers.grb.grb_energy import ICONGrbSolver
from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.method.Solvers.grb.grb_qpsolver import QPGrbSolver
from openpto.method.Solvers.grb.grb_tsp import TSPGrbSolver
from openpto.method.Solvers.heuristic.dp import DPSolver
from openpto.method.Solvers.heuristic.lkh import LKHSolver
from openpto.method.Solvers.heuristic.spSolver import spSolver
from openpto.method.Solvers.heuristic.TopKSolver import TopKSolver
from openpto.method.Solvers.neural.BudgetallocSolver import budgetallocSolver
from openpto.method.Solvers.neural.softTopkSolver import softTopkSolver
from openpto.method.Solvers.ortools.ortools_ad import AdOrToolSolver


################################# Wrappers ################################################
def solver_wrapper(args, conf, problem):
    prob_solver_dict = {
        "budgetalloc": {"neural": budgetallocSolver},
        "bipartitematching": {"cvxpy": BmatchingSolver},
        "portfolio": {"cvxpy": CpPortfolioSolver},
        "cubic": {"heuristic": TopKSolver, "neural": softTopkSolver},
        "energy": {"gurobi": ICONGrbSolver},
        "knapsack": {
            "gurobi": KPGrbSolver,
            "heuristic": DPSolver,
            "qptl": QPGrbSolver,
            "cvxpy": CpKPSolver,
        },
        "advertising": {"gurobi": AdGrbSolver, "ortools": AdOrToolSolver},
        "shortestpath": {"heuristic": spSolver},
        "TSP": {"gurobi": TSPGrbSolver, "heuristic": LKHSolver},
    }
    solve_dict = {**problem.init_API(), **conf["solver"][args.solver]}
    return prob_solver_dict[args.problem][args.solver](**solve_dict)
