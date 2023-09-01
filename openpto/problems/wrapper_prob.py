import os
import pickle

from functools import partial
from typing import Dict

import pandas as pd

from .BipartiteMatching import BipartiteMatching
from .BudgetAllocation import BudgetAllocation
from .CubicTopK import CubicTopK
from .Knapsack import Knapsack
from .PortfolioOpt import PortfolioOpt
from .RMAB import RMAB


################################# Wrappers ################################################
def problem_wrapper(args, conf):
    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
    ProblemClass = str2prob(args.problem)
    problemKwargs = prob2args(args, conf)
    problem = init_problem(ProblemClass, problemKwargs)
    return problem


def str2prob(prob_str):
    prob_dict = {
        "budgetalloc": BudgetAllocation,
        "cubic": CubicTopK,
        "bipartitematching": BipartiteMatching,
        "rmab": RMAB,
        "portfolio": PortfolioOpt,
        "knapsack": Knapsack,
    }
    # TODO: more problems
    return prob_dict[prob_str]


def prob2args(args, conf):
    common_kwargs = {
        "data_dir": args.data_dir,
        "num_train_instances": args.instances,
        "num_test_instances": args.testinstances,
        "val_frac": args.valfrac,
        "rand_seed": args.seed,
    }
    # if args.problem == 'budgetalloc':
    #     # assert args.opt_model in ['']
    #     problem_kwargs =    {}
    # elif args.problem == 'cubic':
    #     # assert args.opt_model in ['']
    #     problem_kwargs =    {}
    # elif args.problem == 'bipartitematching':
    #     # assert args.opt_model in ['']
    #     problem_kwargs =    {}
    # elif args.problem == 'rmab':
    #     # assert args.opt_model in ['']
    #     problem_kwargs =    {}
    # elif args.problem == 'portfolio':
    #     # assert args.opt_model in ['']
    #     problem_kwargs =    {}
    # else:
    problem_kwargs = {}
    return {**conf["dataset"], **common_kwargs, **problem_kwargs}


def init_if_not_saved(
    problem_cls,
    kwargs,
    folder="saved_problems",
    load_new=True,
):
    # Find the filename if a saved version of the problem with the same kwargs exists
    master_filename = os.path.join(folder, f"{problem_cls.__name__}.csv")
    filename, saved_probs = find_saved_problem(master_filename, kwargs)

    if not load_new and filename is not None:
        # Load the model
        with open(filename, "rb") as file:
            problem = pickle.load(file)
    else:
        # Initialise model from scratch
        problem = problem_cls(**kwargs)

        # Save model for the future
        print("Saving the problem")
        filename = os.path.join(folder, f"{problem_cls.__name__}_{len(saved_probs)}.pkl")
        with open(filename, "wb") as file:
            pickle.dump(problem, file)

        # Add its details to the master file
        kwargs["filename"] = filename
        # saved_probs = saved_probs.append([kwargs])
        saved_probs = pd.concat([saved_probs, pd.DataFrame([kwargs])], ignore_index=True)
        with open(master_filename, "w") as file:
            saved_probs.to_csv(file, index=False)

    return problem


def find_saved_problem(
    master_filename: str,
    kwargs: Dict,
):
    # Open the master file with details about saved models
    if os.path.exists(master_filename):
        with open(master_filename, "r") as file:
            saved_probs = pd.read_csv(file)
    else:
        saved_probs = pd.DataFrame(
            columns=(
                "filename",
                *kwargs.keys(),
            )
        )

    # Check if the problem has been saved before
    relevant_models = saved_probs
    for col, val in kwargs.items():
        if col in relevant_models.columns:
            relevant_models = relevant_models.loc[
                relevant_models[col] == val
            ] # filtering models by parameters

    # If it has, find the relevant filename
    filename = None
    if not relevant_models.empty:
        filename = relevant_models["filename"].values[0]

    return filename, saved_probs
