import numpy as np
import torch

from openpto.method.Models.abcOptModel import optModel


class MSE(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, **kwargs):
        super().__init__(optSolver, processes, solve_ratio)

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
        loss = (coeff_hat - coeff_true).square()
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            pass
        else:
            raise ValueError("No reduction '{}'.".format(hyperparams["reduction"]))
        return loss


class MAE(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, **kwargs):
        super().__init__(optSolver, processes, solve_ratio, **kwargs)

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
        loss = (coeff_hat - coeff_true).abs()
        if hyperparams["reduction"] == "mean":
            loss = torch.mean(loss)
        elif hyperparams["reduction"] == "sum":
            loss = torch.sum(loss)
        elif hyperparams["reduction"] == "none":
            pass
        else:
            raise ValueError("No reduction '{}'.".format(hyperparams["reduction"]))
        return loss


class CE(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, **kwargs):
        super().__init__(optSolver, processes, solve_ratio, **kwargs)

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
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, **kwargs):
        super().__init__(optSolver, processes, solve_ratio, **kwargs)

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
        sum_loss = (coeff_hat - coeff_true).sum(dim=-1).square()  # .mean()
        loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(coeff_hat, coeff_true)
        return loss_regularised


class DFL(optModel):
    def __init__(self, optSolver=None, processes=1, solve_ratio=1, **kwargs):
        super().__init__(optSolver, processes, solve_ratio, **kwargs)
        self.dflalpha = kwargs["dflalpha"]

    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        if problem.get_twostageloss() == "mse":
            twostageloss = MSE()
        elif problem.get_twostageloss() == "ce":
            twostageloss = CE()
        else:
            raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")
        sol_hat, objs = problem.get_decision(
            coeff_hat,
            params=params,
            optSolver=self.optSolver,
            isTrain=True,
            **problem.init_API(),
        )
        if isinstance(objs, np.ndarray):
            objs = torch.from_numpy(objs.astype("float")).to(problem.device)
        twostage_loss = twostageloss(problem, coeff_hat, coeff_true)
        loss = -objs + self.dflalpha * twostage_loss
        return loss
