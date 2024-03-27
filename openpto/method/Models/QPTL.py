#!/usr/bin/env python
# coding: utf-8
"""
"""

from gurobipy import GRB

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_tensor


class QPTL(optModel):
    """ """

    def __init__(self, ptoSolver, tau=1, **kwargs):
        """ """
        super().__init__(ptoSolver)
        self.tau = tau

    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true,
        params,
        **hyperparams,
    ):
        """
        Forward pass
        """
        # n_items = coeff_true.shape[1]
        # Q = torch.eye(n_items) / hyperparams["tau"]
        # # G = torch.cat((torch.from_numpy(weights).float(), torch.diagflat(torch.ones(n_items)),
        # # torch.diagflat(torch.ones(n_items)*-1)), 0)
        # # h = torch.cat((torch.tensor([capacity],dtype=torch.float),torch.ones(n_items),torch.zeros(n_items)))

        # G = torch.from_numpy(weights).float()
        # h = torch.tensor([capacity], dtype=torch.float)

        # c_true = -coeff_true
        # c_pred = -coeff_hat
        # solver = QPFunction(
        #     verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad
        # )
        # x = solver(
        #     Q.expand(n_train, *Q.shape),
        #     c_pred.squeeze(),
        #     G.expand(n_train, *G.shape),
        #     h.expand(n_train, *h.shape),
        #     torch.Tensor(),
        #     torch.Tensor(),
        # )
        # loss = (x.squeeze() * c_true).mean()

        # get device
        device = coeff_hat.device
        # coeff_hat = coeff_hat.squeeze(-1)

        # get true solution
        sol_true, _ = problem.get_decision(
            coeff_true,
            params=params,
            ptoSolver=self.ptoSolver,
            isTrain=False,
            **problem.init_API(),
        )
        sol_true = to_tensor(sol_true).to(device)

        obj_cp = problem.get_objective(coeff_hat, sol_true)

        # get loss
        if self.ptoSolver.modelSense == GRB.MINIMIZE:
            loss = obj_cp
        elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
            loss = -obj_cp
        else:
            raise NotImplementedError
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss
