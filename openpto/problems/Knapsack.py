import os
import itertools
import quandl
import datetime as dt
import random
import pdb

import pandas as pd

import torch

from openpto.problems.PTOProblem import PTOProblem

class Knapsack(PTOProblem):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_items=100,  # number of targets to consider
        num_fake_targets=5000,  # number of random features added to make the task harder
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(Knapsack, self).__init__()
        self._set_seed(rand_seed)
    
    @staticmethod
    def gendata():
        return