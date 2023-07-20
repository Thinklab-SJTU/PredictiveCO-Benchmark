

from abc import abstractmethod
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool

import numpy as np
from torch import nn

class optModel(nn.Module):
    """
        An abstract module for the learning to rank losses, which measure the difference in how the predicted cost
        vector and the true cost vector rank a pool of feasible solutions.
    """
    def __init__(self, optSolver, processes=1, solve_ratio=1, dataset=None):
        return

    @abstractmethod
    def forward(self, coeff_hat, coeff_true=None, sol_hat=None, sol_true=None, params=None):
        """
            Input:
                coeff_hat:
                coeff_true:
                sol_hat:
                sol_true:
                params:
            Output: 
                sol, obj, loss
        """

        pass