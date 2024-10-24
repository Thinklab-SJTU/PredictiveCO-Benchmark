import torch

from ortools.linear_solver import pywraplp  # pylint: disable=no-name-in-module

from openpto.method.Solvers.abcptoSolver import ptoSolver
from openpto.method.utils_method import to_array


# optimization model
class AdOrToolSolver(ptoSolver):
    def __init__(self, modelSense, **kwargs):
        super().__init__(modelSense)

    def solve(self, profits, cost_pv, given_pv):
        return