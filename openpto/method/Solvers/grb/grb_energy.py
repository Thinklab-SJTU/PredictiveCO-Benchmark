import gurobipy as gp  # pylint: disable=no-name-in-module
import numpy as np

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Solvers.grb.grbSolver import optGrbSolver


# optimization model
class ICONGrbSolver(optGrbSolver):
    def __init__(
        self,
    ):
        return

    def _getModel(self):
        return

    def solve(self, price, timelimit=None):
        return
