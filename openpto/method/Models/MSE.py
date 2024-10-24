import torch
import torch.nn as nn

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_device, to_tensor


class MSE(optModel):
    def __init__(self, ptoSolver=None, **kwargs):
        super().__init__(ptoSolver)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        loss = 0
        return loss


class MAE(optModel):
    def __init__(self, ptoSolver=None, **kwargs):
        super().__init__(ptoSolver, **kwargs)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        loss = 0
        return loss


class BCE(optModel):
    def __init__(self, ptoSolver=None, **kwargs):
        super().__init__(ptoSolver, **kwargs)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        loss = 0
        return loss



class CE(optModel):
    def __init__(self, ptoSolver=None, **kwargs):
        super().__init__(ptoSolver, **kwargs)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        loss = 0
        return loss


class MSE_Sum(optModel):
    def __init__(self, ptoSolver=None, **kwargs):
        super().__init__(ptoSolver, **kwargs)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        loss = 0
        return loss


class DFL(optModel):
    def __init__(self, ptoSolver=None, **kwargs):
        super().__init__(ptoSolver, **kwargs)
        self.dflalpha = kwargs["dflalpha"]

    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        loss = 0
        return loss
