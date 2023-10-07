import numpy as np
import torch

from openpto.method.utils_method import move_to_array


def get_eval_results(problem, coeff_true, sols_true, sols_hat, aux_data):
    if problem.get_eval_metric() == "regret":
        return regret_func(problem, coeff_true, sols_true, sols_hat)
    elif problem.get_eval_metric() == "treatment":
        return treatment_func(aux_data, coeff_true)
    else:
        raise NotImplementedError("Not implemented")


def regret_func(problem, coeff_true, sols_true, sols_hat):
    if torch.is_tensor(coeff_true):
        coeff_true = coeff_true.detach().cpu()
    objs_hat = problem.get_objective(coeff_true, sols_hat)
    objs_true = problem.get_objective(coeff_true, sols_true)
    regret = abs(objs_hat - objs_true)
    regret = move_to_array(regret)
    return {"value": regret}


def treatment_func(aux_data, labels):
    aux_data, labels = aux_data.cpu(), labels.cpu()
    n_instances = aux_data.shape[0]
    ctr_treats, ctr_controls = [], []
    for idx in range(n_instances):
        label_idx = labels[idx].squeeze(-1)
        realchannels, mockchannels = aux_data[idx, 0, :], aux_data[idx, 1, :]
        # calculate ctr
        treat_mask = realchannels == mockchannels
        treat_label = label_idx[treat_mask]
        control_label = label_idx[~treat_mask]
        ctr_treat = sum(treat_label) / len(treat_label)
        ctr_control = sum(control_label) / len(control_label)
        # collect
        ctr_treats.append(ctr_treat)
        ctr_controls.append(ctr_control)
    return {
        "ctr_treat": np.mean(ctr_treats),
        "ctr_control": np.mean(ctr_controls),
        "value": np.mean(ctr_treats) - np.mean(ctr_controls),
    }
