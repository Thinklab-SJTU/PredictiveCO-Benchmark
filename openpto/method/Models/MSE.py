import torch

from openpto.method.Models.abcOptModel import optModel


class MSE(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, dataset=None):
        super().__init__(optSolver, processes, solve_ratio, dataset)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        """
        Calculates the mean squared error between predictions
        Yhat and true lables Y.
        """
        return (coeff_hat - coeff_true).square().mean()


class MAE(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, dataset=None):
        super().__init__(optSolver, processes, solve_ratio, dataset)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        """
        Calculates the mean squared error between predictions
        Yhat and true lables Y.
        """
        return (coeff_hat - coeff_true).abs().mean()


class CE(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, dataset=None):
        super().__init__(optSolver, processes, solve_ratio, dataset)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        return torch.nn.BCELoss()(coeff_hat, coeff_true)


class MSE_Sum(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, dataset=None):
        super().__init__(optSolver, processes, solve_ratio, dataset)

    @staticmethod
    def forward(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        """
            Custom loss function that the squared error of the _sum_
            along the last dimension plus some regularisation.
            Useful for the Submodular Optimisation problems in Wilder et. al.
        Input:
            alpha:  #weight of MSE-based regularisation
        """
        # Check if prediction is a matrix/tensor
        assert len(coeff_true.shape) >= 2
        alpha = hyperparams["alpha"]

        # Calculate loss
        sum_loss = (coeff_hat - coeff_true).sum(dim=-1).square().mean()
        loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(coeff_hat, coeff_true)
        return loss_regularised
