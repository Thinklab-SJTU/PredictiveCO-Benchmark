import torch

from openpto.method.utils_method import move_to_array


def regret_func(problem, coeff_true, sols_true, sols_hat):
    if torch.is_tensor(coeff_true):
        coeff_true = coeff_true.detach().cpu().numpy()
    objs_hat = problem.get_objective(coeff_true, sols_hat)
    objs_true = problem.get_objective(coeff_true, sols_true)
    regret = abs(objs_hat - objs_true)
    regret = move_to_array(regret)
    return regret
