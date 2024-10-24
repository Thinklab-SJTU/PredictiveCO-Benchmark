import json
import os
import pickle

from functools import partial
from typing import Dict

import pandas as pd

from openpto.problems.Advertising import Advertising
from openpto.problems.BipartiteMatching import BipartiteMatching
from openpto.problems.BudgetAllocation import BudgetAllocation
from openpto.problems.CubicTopK import CubicTopK
from openpto.problems.Energy import Energy
from openpto.problems.Knapsack import Knapsack
from openpto.problems.PortfolioOpt import PortfolioOpt
from openpto.problems.Shortestpath import Shortestpath
from openpto.problems.TSP import TSP


################################# Wrappers ################################################
def problem_wrapper(args, conf):
    init_problem = partial(
        init_if_not_saved, folder="saved_problems", load_new=args.loadnew
    )
    ProblemClass = str2prob(args.problem)
    problemKwargs = prob2args(args, conf)
    problem = init_problem(ProblemClass, problemKwargs)
    return problem


def str2prob(prob_str):
    prob_dict = {
        "budgetalloc": BudgetAllocation,
        "cubic": CubicTopK,
        "bipartitematching": BipartiteMatching,
        "portfolio": PortfolioOpt,
        "knapsack": Knapsack,
        "energy": Energy,
        "advertising": Advertising,
        "shortestpath": Shortestpath,
        "TSP": TSP,
    }
    return prob_dict[prob_str]


def prob2args(args, conf):
    common_kwargs = {
        "data_dir": args.data_dir,
        "num_train_instances": args.instances,
        "num_test_instances": args.testinstances,
        "rand_seed": args.seed,
    }
    problem_kwargs = {}
    return {**conf["dataset"], **common_kwargs, **problem_kwargs}


def init_if_not_saved(
    problem_cls,
    kwargs,
    folder="saved_problems",
    load_new=True,
):
    # Find the filename if a saved version of the problem with the same kwargs exists
    prob_name = problem_cls.__name__
    os.makedirs(os.path.join(folder, prob_name), exist_ok=True)
    master_filename = os.path.join(folder, prob_name, f"{prob_name}.csv")
    filename, saved_probs = find_saved_problem(master_filename, kwargs)

    if not load_new and filename is not None:
        print("\n--------loading:", filename, "\n")
        # Load the model
        with open(filename, "rb") as file:
            problem = pickle.load(file)
    else:
        # Initialise model from scratch
        print("\n--------Init new problem with kwargs:", kwargs)
        problem = problem_cls(**kwargs)

        # Save model for the future
        filename = os.path.join(folder, prob_name, f"{prob_name}_{len(saved_probs)}.pkl")
        print(f"Saving the problem to {filename}\n")
        with open(filename, "wb") as file:
            pickle.dump(problem, file)

        # Add its details to the master file
        kwargs["filename"] = filename
        saved_probs = pd.concat([saved_probs, pd.DataFrame([kwargs])], ignore_index=True)
        with open(master_filename, "w") as file:
            saved_probs.to_csv(file, index=False)

    return problem


def dict_to_string(d):
    if isinstance(d, dict):
        return json.dumps(d)
    else:
        return d


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
        if isinstance(val, dict):
            relevant_models = relevant_models  # TODO: work around
            continue
        if col in relevant_models.columns:
            relevant_models = relevant_models.loc[
                relevant_models[col] == val
            ]  # filtering models by parameters
    # If it has, find the relevant filename
    filename = None
    if not relevant_models.empty:
        filename = relevant_models["filename"].values[-1]
    return filename, saved_probs
