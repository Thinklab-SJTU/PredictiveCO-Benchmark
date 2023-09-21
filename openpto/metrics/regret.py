import torch


def regret_func(problem, coeff_true, sols_true, sols_hat):
    if torch.is_tensor(coeff_true):
        coeff_true = coeff_true.detach().cpu().numpy()
    objs_hat = problem.get_objective(coeff_true, sols_hat)
    objs_true = problem.get_objective(coeff_true, sols_true)
    regret = abs(objs_hat - objs_true)
    if torch.is_tensor(regret):  # convert to tensor
        regret = regret.cpu().numpy()
    return regret
