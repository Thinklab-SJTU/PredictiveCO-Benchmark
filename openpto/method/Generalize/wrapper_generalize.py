from openpto.method.Generalize.EERM import EERM, ERM


################################ Wrappers ################################################
def generalize_wrapper(args, ood_name, pred_model):
    ood_dict = {"ERM": ERM, "EERM": EERM}
    return ood_dict[ood_name](pred_model, args.n_envs, args.alpha, args.beta)
