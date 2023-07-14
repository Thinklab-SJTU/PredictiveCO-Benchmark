import os
from typing import Dict

import pickle
import pandas as pd

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


def init_if_not_saved(
    problem_cls,
    kwargs,
    folder='saved_models',
    load_new=True,
):
    # Find the filename if a saved version of the problem with the same kwargs exists
    master_filename = os.path.join(folder, f"{problem_cls.__name__}.csv")
    filename, saved_probs = find_saved_problem(master_filename, kwargs)
 
    if not load_new and filename is not None:
        # Load the model
        with open(filename, 'rb') as file:
            problem = pickle.load(file)
    else:
        # Initialise model from scratch
        problem = problem_cls(**kwargs)

        # Save model for the future
        print("Saving the problem")
        filename = os.path.join(folder, f"{problem_cls.__name__}_{len(saved_probs)}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(problem, file)

        # Add its details to the master file
        kwargs['filename'] = filename
        # saved_probs = saved_probs.append([kwargs])
        saved_probs = pd.concat([saved_probs, pd.DataFrame([kwargs])], ignore_index=True)
        with open(master_filename, 'w') as file:
            saved_probs.to_csv(file, index=False)

    return problem

def find_saved_problem(
    master_filename: str,
    kwargs: Dict,
):
    # Open the master file with details about saved models
    if os.path.exists(master_filename):
        with open(master_filename, 'r') as file:
            saved_probs = pd.read_csv(file)
    else:
        saved_probs = pd.DataFrame(columns=('filename', *kwargs.keys(),))
    
    # Check if the problem has been saved before
    relevant_models = saved_probs
    for col, val in kwargs.items():
        if col in relevant_models.columns:
            relevant_models = relevant_models.loc[relevant_models[col] == val]  # filtering models by parameters

    # If it has, find the relevant filename
    filename = None
    if not relevant_models.empty:
        filename = relevant_models['filename'].values[0]
    
    return filename, saved_probs
