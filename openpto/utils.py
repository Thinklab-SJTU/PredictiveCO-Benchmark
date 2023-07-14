import argparse
import ast
import os
import pdb
import inspect
from itertools import repeat

import pickle
import pandas as pd

import torch

def get_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--problem', type=str, choices=['budgetalloc', 'bipartitematching', 
                                    'cubic', 'rmab', 'portfolio', 'Knapsack','Energy'], default='portfolio')
    parser.add_argument('--method', type=str, default='SPO', choices=['SPO'], help="Select methods")
    parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")

    # training
    parser.add_argument('--loadnew', type=ast.literal_eval, default=False)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=1000)
    # data
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--instances', type=int, default=400)
    parser.add_argument('--testinstances', type=int, default=200)
    # debug
    parser.add_argument('--valfrac', type=float, default=0.5)
    parser.add_argument('--valfreq', type=int, default=5)

    # model
    parser.add_argument('--model', type=str, choices=['dense'], default='dense')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--loss', type=str, choices=['mse', 'msesum', 'dense', 'weightedmse', 'weightedmse++', 'weightedce', 'weightedmsesum', 'dfl', 'quad', 'quad++', 'ce'], default='mse')

    #   Domain-specific: BudgetAllocation or CubicTopK
    parser.add_argument('--budget', type=int, default=1)
    parser.add_argument('--numitems', type=int, default=50)
    #   Domain-specific: BudgetAllocation
    parser.add_argument('--numtargets', type=int, default=10)
    parser.add_argument('--fakefeatures', type=int, default=0)
    #   Domain-specific: RMAB
    parser.add_argument('--rmabbudget', type=int, default=1)
    parser.add_argument('--numarms', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--minlift', type=float, default=0.2)
    parser.add_argument('--scramblinglayers', type=int, default=3)
    parser.add_argument('--scramblingsize', type=int, default=64)
    parser.add_argument('--numfeatures', type=int, default=16)
    parser.add_argument('--noisestd', type=float, default=0.5)
    parser.add_argument('--eval', type=str, choices=['exact', 'sim'], default='exact')
    #   Domain-specific: BipartiteMatching
    parser.add_argument('--nodes', type=int, default=10)
    #   Domain-specific: PortfolioOptimization
    parser.add_argument('--stocks', type=int, default=50)
    parser.add_argument('--stockalpha', type=float, default=0.1)
    #   Decision-Focused Learning
    parser.add_argument('--dflalpha', type=float, default=1.)
    #   Learned-Loss
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--sampling', type=str, choices=['random', 'random_flip', 'random_uniform', 'numerical_jacobian', 'random_jacobian', 'random_hessian', 'random'], default='random')
    parser.add_argument('--samplingstd', type=float)
    parser.add_argument('--numsamples', type=int, default=5000)
    parser.add_argument('--losslr', type=float, default=0.01)
    #       Approach-Specific: Quadratic
    parser.add_argument('--quadrank', type=int, default=20)
    parser.add_argument('--quadalpha', type=float, default=0)
    args = parser.parse_args()
    return args
