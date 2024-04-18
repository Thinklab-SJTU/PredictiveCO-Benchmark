from openpto.method.Generalize.EERM import EERM, ERM


################################ Wrappers ################################################
def generalize_wrapper(args, ood_name, pred_model, logger):
    ood_dict = {"ERM": ERM, "EERM": EERM}
    return ood_dict[ood_name](
        pred_model,
        logger=logger,
        n_envs=args.n_envs,
        alpha=args.alpha,
        beta=args.beta,
        l1_weight=args.l1_weight,
        l2_weight=args.l2_weight,
        ood_reduction=args.ood_reduction,
        use_train=args.use_train,
    )
