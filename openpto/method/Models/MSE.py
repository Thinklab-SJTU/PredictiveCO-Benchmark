import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import to_tensor


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


class BCE(optModel):
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
        if torch.is_tensor(coeff_true):
            coeff_true = coeff_true.float()
            return torch.nn.BCELoss(reduction=hyperparams["reduction"])(
                coeff_hat, coeff_true
            )
        elif isinstance(coeff_true, list):
            loss_list = list()
            for Y_idx in range(len(coeff_true)):
                loss_list.append(torch.nn.BCELoss()(coeff_hat[Y_idx], coeff_true[Y_idx]))
            loss = torch.stack(loss_list)
            if hyperparams["reduction"] == "mean":
                loss = torch.mean(loss)
            elif hyperparams["reduction"] == "sum":
                loss = torch.sum(loss)
            elif hyperparams["reduction"] == "none":
                pass
            else:
                raise ValueError("No reduction '{}'.".format(hyperparams["reduction"]))
        else:
            raise ValueError("coeff_true is not a tensor or list")


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
        return torch.nn.CrossEntropyLoss(reduction=hyperparams["reduction"])(
            coeff_hat, coeff_true
        )


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
        elif problem.get_twostageloss() == "bce":
            twostageloss = BCE()
        elif problem.get_twostageloss() == "ce":
            twostageloss = CE()
        else:
            raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")
        sol_hat, _ = problem.get_decision(
            coeff_hat,
            params=params,
            optSolver=self.optSolver,
            isTrain=True,
            **problem.init_API(),
        )

        sol_hat = to_tensor(sol_hat).to(problem.device)
        obj_hat = problem.get_objective(coeff_hat, sol_hat, params, **problem.init_API()).to(problem.device)
        # loss
        twostage_loss = twostageloss(problem, coeff_hat, coeff_true, **hyperparams)
        if self.optSolver.modelSense == GRB.MINIMIZE:
            loss = obj_hat + self.dflalpha * twostage_loss
        elif self.optSolver.modelSense == GRB.MAXIMIZE:
            loss = -obj_hat + self.dflalpha * twostage_loss
        else:
            raise ValueError(f"Unknown model sense {self.optSolver.modelSense}")
        # debug
        if (
            hyperparams["do_debug"]
            and hyperparams["partition"] == "train"
            and loss.requires_grad
        ):
            objs_hat_grad = torch.autograd.grad(loss, obj_hat, retain_graph=True)
            coeff_hat_grad = torch.autograd.grad(loss, coeff_hat, retain_graph=True)
            twostage_grad = torch.autograd.grad(loss, twostage_loss, retain_graph=True)

            def hook_fn(grad):
                print("gradient through the path:", grad)

            # coeff_hat_grad[0].register_hook(hook_fn)
            # sols_hat_grad = torch.autograd.grad(loss, sol_hat, retain_graph=True)[0]
            print(
                "dbb grad: ",
                objs_hat_grad[0].shape,
                coeff_hat_grad[0].shape,
                twostage_grad,
            )
        return loss
