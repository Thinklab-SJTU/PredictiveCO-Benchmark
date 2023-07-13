from .BudgetAllocation import BudgetAllocation
from .BipartiteMatching import BipartiteMatching
from .PortfolioOpt import PortfolioOpt
from .RMAB import RMAB
from .CubicTopK import CubicTopK

def str2prob(prob_str):
    prob_dict = {"budgetalloc": BudgetAllocation,
                 "cubic":CubicTopK,
                 "bipartitematching": BipartiteMatching,
                 'rmab':RMAB,
                 'portfolio':PortfolioOpt,
                 }
    # TODO: more problems
    return prob_dict[prob_str]