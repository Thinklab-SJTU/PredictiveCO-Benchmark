import numpy as np
import torch

from openpto.method.utils_method import ndiv, to_array


def get_eval_results(problem, coeff_true, sols_true, sols_hat, aux_data):
    if problem.get_eval_metric() == "regret":
        return regret_func(problem, coeff_true, sols_true, sols_hat, aux_data)
    elif problem.get_eval_metric() == "uplift":
        return uplift_func(coeff_true, sols_hat, aux_data)
    elif problem.get_eval_metric() == "match":
        return perfect_match_accuracy(sols_true, sols_hat)
    else:
        raise NotImplementedError("Not implemented")


def regret_func(problem, coeff_true, sols_true, sols_hat, aux_data):
    if torch.is_tensor(coeff_true):
        coeff_true = coeff_true.detach().cpu()
    objs_hat = problem.get_objective(coeff_true, sols_hat, aux_data)
    objs_true = problem.get_objective(coeff_true, sols_true, aux_data)
    regret = abs(objs_hat - objs_true)
    regret = to_array(regret)
    return {"value": regret, "sense": 1}


def perfect_match_accuracy(sols_true, sols_hat):
    # sols_true, sols_hat = to_array(sols_true), to_array(sols_hat)
    accuracy = (sols_hat * sols_true).sum(-1) / sols_true.sum(-1)
    # matching_correct = torch.sum(torch.abs(sols_true - sols_hat), dim=-1)
    # avg_matching_correct = (matching_correct < 0.5).float()
    return {"value": accuracy, "sense": -1}


def uplift_func(labels, sols_hat, aux_data):
    aux_data, labels = to_array(aux_data), to_array(labels)
    n_instances = len(aux_data)
    ctr_treats, ctr_controls = [], []
    # print("sols_hat shape: ", sols_hat.shape, n_instances)
    for idx in range(n_instances):
        label_idx = (
            labels[idx].squeeze(-1).reshape(-1, 4)[:, 0]
        )  # remove duplicated labels
        realchannels = aux_data[idx]
        mockchannels = sol2channel(sols_hat[idx])
        # print("realchannels shape: ", realchannels.shape, mockchannels.shape)
        # calculate ctr
        # print("label_idx: ", label_idx.shape)
        treat_mask = realchannels == mockchannels
        treat_label = label_idx[treat_mask]
        control_label = label_idx[~treat_mask]
        # print("treat_label: ", treat_label.shape, control_label.shape)
        ctr_treat = ndiv(sum(treat_label), len(treat_label))
        ctr_control = ndiv(sum(control_label), len(control_label))
        # print("len(treat_label): ", sum(treat_label), len(treat_label))
        # print("len(control_label): ", sum(control_label), len(control_label))
        # collect
        ctr_treats.append(ctr_treat)
        ctr_controls.append(ctr_control)
    # treats_mean = np.mean(ctr_treats, keepdims=True)
    # controls_mean = np.mean(ctr_controls, keepdims=True)
    ctr_treats, ctr_controls = np.array(ctr_treats), np.array(ctr_controls)
    return {
        "sense": -1,
        "ctr_treat": ctr_treats,
        "ctr_control": ctr_treats,
        "value": ctr_treats - ctr_controls,
    }


def sol2channel(x):
    x = x.reshape(-1, 4)
    # TODO: support muliple instances
    x.shape[0]
    num_workers = x.shape[0]
    mockchannels = []
    for worker in range(num_workers):
        task = np.argmax(x[worker, :])
        mockchannels.append(task)
    return np.array(mockchannels)
