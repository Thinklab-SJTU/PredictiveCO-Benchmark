from openpto.method.Models.MSE import BCE, CE, MAE, MSE


def str2twoStageLoss(problem):
    if problem.get_twostageloss() == "mse":
        twostageloss = MSE()
    elif problem.get_twostageloss() == "bce":
        twostageloss = BCE()
    elif problem.get_twostageloss() == "ce":
        twostageloss = CE()
    elif problem.get_twostageloss() == "mae":
        twostageloss = MAE()
    else:
        raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")
    return twostageloss


def l1_penalty(pred_model):
    return sum([(param.abs()).sum() for param in pred_model.parameters()])


def l2_penalty(pred_model):
    return sum([(param.square()).sum() for param in pred_model.parameters()])
