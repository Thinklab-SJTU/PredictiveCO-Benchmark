import numpy as np
import torch


class OptimiseSubmodular(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(
        ctx,
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        return
