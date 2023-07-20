import random
import os
import time

import torch

import pickle
import matplotlib.pyplot as plt

from openpto.method.models import _get_learned_loss, SPOPlus

NUM_CPUS = os.cpu_count()

def get_loss_fn(
    name,
    problem,
    **kwargs
):
    if name == 'mse':
        return MSE
    elif name == 'msesum':
        return MSE_Sum
    elif name == 'ce':
        return CE
    elif name == 'dfl':
        return _get_decision_focused(problem, **kwargs)
    elif name == 'learned':
        return _get_learned_loss(problem, name, **kwargs)
    elif name == 'SPO':
        return SPOPlus
    elif name == 'LTR':
        return None
    elif name == 'Intopt':
        return None
    elif name == 'NCE':
        return None
    elif name == 'Blackbox':
        return None
    else:
        raise LookupError()


def MSE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).square().mean()


def MAE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).abs().mean()


def CE(Yhats, Ys, **kwargs):
    return torch.nn.BCELoss()(Yhats, Ys)


def MSE_Sum(
    Yhats,
    Ys,
    alpha=0.1,  # weight of MSE-based regularisation
    **kwargs
):
    """
    Custom loss function that the squared error of the _sum_
    along the last dimension plus some regularisation.
    Useful for the Submodular Optimisation problems in Wilder et. al.
    """
    # Check if prediction is a matrix/tensor
    assert len(Ys.shape) >= 2

    # Calculate loss
    sum_loss = (Yhats - Ys).sum(dim=-1).square().mean()
    loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(Yhats, Ys)
    return loss_regularised


def _get_decision_focused(
    problem,
    dflalpha=1.,
    **kwargs,
):
    if problem.get_twostageloss() == 'mse':
        twostageloss = MSE
    elif problem.get_twostageloss() == 'ce':
        twostageloss = CE
    else:
        raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")

    def decision_focused_loss(Yhats, Ys, **kwargs):
        Zs = problem.get_decision(Yhats, isTrain=True, **kwargs)
        obj = problem.get_objective(Ys, Zs, isTrain=True, **kwargs)
        loss = -obj + dflalpha * twostageloss(Yhats, Ys)

        return loss

    return decision_focused_loss



