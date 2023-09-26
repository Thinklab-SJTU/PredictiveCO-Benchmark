from openpto.method.Models.MSE import CE, MAE, MSE


def str2twoStageLoss(problem):
    if problem.get_twostageloss() == "mse":
        twostageloss = MSE()
    elif problem.get_twostageloss() == "ce":
        twostageloss = CE()
    elif problem.get_twostageloss() == "mae":
        twostageloss = MAE()
    else:
        raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")
    return twostageloss
