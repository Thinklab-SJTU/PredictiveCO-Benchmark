from abc import abstractmethod

from torch import nn


class optModel(nn.Module):
    """ """

    def __init__(self, ptoSolver=None, **kwargs):
        super(optModel, self).__init__()
        self.ptoSolver = ptoSolver

    @abstractmethod
    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true=None,
        params=None,
        **hyperparams,
    ):
        """
        Input:
            problem:
            coeff_hat:
            coeff_true:
            params:
        Output:
            sol, obj, loss
        """

        pass
