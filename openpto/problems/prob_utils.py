import os
from typing import Dict
from functools import partial

import pickle
import pandas as pd

from .BudgetAllocation import BudgetAllocation
from .BipartiteMatching import BipartiteMatching
from .PortfolioOpt import PortfolioOpt
from .RMAB import RMAB
from .CubicTopK import CubicTopK

def problem_wrapper(args):
    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
    ProblemClass = str2prob(args.problem)
    problemKwargs = prob2args(args)
    problem = init_problem(ProblemClass, problemKwargs)
    return problem

def str2prob(prob_str):
    prob_dict = {"budgetalloc": BudgetAllocation,
                 "cubic":CubicTopK,
                 "bipartitematching": BipartiteMatching,
                 'rmab':RMAB,
                 'portfolio':PortfolioOpt,
                 }
    # TODO: more problems
    return prob_dict[prob_str]

def prob2args(args):
    if args.problem == 'budgetalloc':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_targets': args.numtargets,
                            'num_items': args.numitems,
                            'budget': args.budget,
                            'num_fake_targets': args.fakefeatures,
                            'rand_seed': args.seed,
                            'val_frac': args.valfrac,}
    elif args.problem == 'cubic':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_items': args.numitems,
                            'budget': args.budget,
                            'rand_seed': args.seed,
                            'val_frac': args.valfrac,}
    elif args.problem == 'bipartitematching':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_nodes': args.nodes,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
    elif args.problem == 'rmab':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_arms': args.numarms,
                            'eval_method': args.eval,
                            'min_lift': args.minlift,
                            'budget': args.rmabbudget,
                            'gamma': args.gamma,
                            'num_features': args.numfeatures,
                            'num_intermediate': args.scramblingsize,
                            'num_layers': args.scramblinglayers,
                            'noise_std': args.noisestd,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
    elif args.problem == 'portfolio':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_stocks': args.stocks,
                            'alpha': args.stockalpha,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
    else:
        raise NotImplementedError
    return problem_kwargs

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
