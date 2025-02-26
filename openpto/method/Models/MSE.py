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
        """
        Calculates the mean squared error between predictions
        Yhat and true lables Y.
        """
        loss = (coeff_hat - coeff_true).square()
        # print("(coeff_hat - coeff_true): ", (coeff_hat - coeff_true)[0])
        # print("loss: ", loss[0])
        loss = do_reduction(loss, hyperparams["reduction"])
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
        """
        Calculates the mean squared error between predictions
        Yhat and true lables Y.
        """
        loss = (coeff_hat - coeff_true).abs()
        loss = do_reduction(loss, hyperparams["reduction"])
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
        if torch.is_tensor(coeff_hat):
            coeff_true = coeff_true.float()
            print("coeff_hat:", coeff_hat, "coeff_true: ", coeff_true)
            return nn.BCELoss(reduction=hyperparams["reduction"])(coeff_hat, coeff_true)
        elif isinstance(coeff_hat, list):
            loss_list = list()
            for Y_idx in range(len(coeff_true)):
                loss_list.append(nn.BCELoss()(coeff_hat[Y_idx], coeff_true[Y_idx]))
            loss = torch.stack(loss_list)
            loss = do_reduction(loss, hyperparams["reduction"])
            return loss
        else:
            raise ValueError("coeff_true is not a tensor or list")


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
        return nn.CrossEntropyLoss(reduction=hyperparams["reduction"])(
            coeff_hat, coeff_true
        )


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
        if problem.get_twostageloss() == "mse":
            twostageloss = MSE()
        elif problem.get_twostageloss() == "bce":
            twostageloss = BCE()
        elif problem.get_twostageloss() == "ce":
            twostageloss = CE()
        else:
            raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")
        sol_hat, _ = problem.get_decision(
            coeff_hat,
            params=params,
            ptoSolver=self.ptoSolver,
            isTrain=True,
            **problem.init_API(),
        )

        sol_hat = to_device(to_tensor(sol_hat), problem.device)
        obj_hat = problem.get_objective(
            coeff_hat, sol_hat, params, **problem.init_API()
        ).to(problem.device)
        # loss
        twostage_loss = twostageloss(problem, coeff_hat, coeff_true, **hyperparams)
        if self.ptoSolver.modelSense == GRB.MINIMIZE:
            loss = obj_hat + self.dflalpha * twostage_loss
        elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
            loss = -obj_hat + self.dflalpha * twostage_loss
        else:
            raise ValueError(f"Unknown model sense {self.ptoSolver.modelSense}")
        # debug
        # print("loss: ", loss.shape)
        # assert 0
        if (
            hyperparams["do_debug"]
            and hyperparams["partition"] == "train"
            and loss.requires_grad
        ):
            objs_hat_grad = torch.autograd.grad(loss, obj_hat, retain_graph=True)
            coeff_hat_grad = torch.autograd.grad(loss, coeff_hat, retain_graph=True)
            twostage_grad = torch.autograd.grad(loss, twostage_loss, retain_graph=True)

            print(
                "dbb grad: ",
                objs_hat_grad[0].shape,
                coeff_hat_grad[0].shape,
                twostage_grad,
            )
        print(loss.shape)
        return loss
