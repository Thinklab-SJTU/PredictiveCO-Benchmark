import os

from openpto.method.Models.Blackbox import blackboxOpt
from openpto.method.Models.Identity import negativeIdentity
from openpto.method.Models.LODL import _get_learned_loss
from openpto.method.Models.LTR import listwiseLTR, pairwiseLTR, pointwiseLTR
from openpto.method.Models.MSE import CE, MSE, MSE_Sum
from openpto.method.Models.QPTL import QPTL
from openpto.method.Models.SPO import SPOPlus

# from openpto.method.Models.Intopt import Intopt
# from openpto.method.Models.NCE import NCE


NUM_CPUS = os.cpu_count()


def get_loss_fn(name, problem, **kwargs):
    if name == "mse":
        return MSE
    elif name == "msesum":
        return MSE_Sum
    elif name == "ce":
        return CE
    elif name == "dfl":
        return _get_decision_focused(problem, **kwargs)
    elif name == "learned":
        return _get_learned_loss(problem, name, **kwargs)
    elif name == "spo":
        return SPOPlus
    elif name == "pointLTR":
        return pointwiseLTR
    elif name == "pairLTR":
        return pairwiseLTR
    elif name == "listLTR":
        return listwiseLTR
    elif name == "QPTL":
        return QPTL
    elif name == "intopt":
        return None
    elif name == "nce":
        return None
    elif name == "blackbox":
        return blackboxOpt
    elif name == "identity":
        return negativeIdentity
    else:
        raise LookupError()


# def _get_decision_focused( problem, dflalpha=1., **kwargs,):
def _get_decision_focused(
    problem,
    coeff_hat,
    coeff_true,
    params=None,
    **hyperparams,
):
    dflalpha = hyperparams["dflalpha"]
    if problem.get_twostageloss() == "mse":
        twostageloss = MSE
    elif problem.get_twostageloss() == "ce":
        twostageloss = CE
    else:
        raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")

    # def decision_focused_loss(Yhats, Ys, **kwargs):
    def decision_focused_loss(
        problem,
        coeff_hat,
        coeff_true,
        params=None,
        **hyperparams,
    ):
        sol_hat, _ = problem.get_decision(coeff_hat, isTrain=True, **hyperparams)
        obj = problem.get_objective(coeff_true, sol_hat, isTrain=True, **hyperparams)
        loss = -obj + dflalpha * twostageloss(coeff_hat, coeff_true)
        return loss

    return decision_focused_loss
