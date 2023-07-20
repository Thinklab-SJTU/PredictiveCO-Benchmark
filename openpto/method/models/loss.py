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


def MSE(problem, coeff_hat, coeff_true, sol_hat=None, sol_true=None, params=None, **hyperparams):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (coeff_hat - coeff_true).square().mean()


def MAE(problem, coeff_hat, coeff_true, sol_hat=None, sol_true=None, params=None, **hyperparams):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (coeff_hat - coeff_true).abs().mean()


def CE(problem, coeff_hat, coeff_true, sol_hat=None, sol_true=None, params=None, **hyperparams):
    return torch.nn.BCELoss()(coeff_hat, coeff_true)


# def MSE_Sum(Yhats, Ys, alpha=0.1,  **kwargs):
def MSE_Sum(problem, coeff_hat, coeff_true, sol_hat=None, sol_true=None, params=None, **hyperparams):
    """
        Custom loss function that the squared error of the _sum_
        along the last dimension plus some regularisation.
        Useful for the Submodular Optimisation problems in Wilder et. al.
    Input:
        alpha:  #weight of MSE-based regularisation
    """
    # Check if prediction is a matrix/tensor
    assert len(coeff_true.shape) >= 2
    alpha = hyperparams['alpha']
    
    # Calculate loss
    sum_loss = (coeff_hat - coeff_true).sum(dim=-1).square().mean()
    loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(coeff_hat, coeff_true)
    return loss_regularised


# def _get_decision_focused( problem, dflalpha=1., **kwargs,):
def _get_decision_focused(problem, coeff_hat, coeff_true, sol_hat=None, sol_true=None, params=None, **hyperparams):
    dflalpha = hyperparams['dflalpha']
    if problem.get_twostageloss() == 'mse':
        twostageloss = MSE
    elif problem.get_twostageloss() == 'ce':
        twostageloss = CE
    else:
        raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")

    # def decision_focused_loss(Yhats, Ys, **kwargs):
    def decision_focused_loss(problem, coeff_hat, coeff_true, sol_hat=None, sol_true=None, params=None, **hyperparams):
        Zs = problem.get_decision(coeff_hat, isTrain=True, **hyperparams)
        obj = problem.get_objective(coeff_true, sol_true, isTrain=True, **hyperparams)
        loss = -obj + dflalpha * twostageloss(coeff_hat, coeff_true)

        return loss

    return decision_focused_loss



