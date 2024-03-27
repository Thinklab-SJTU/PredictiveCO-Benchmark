def get_ml_loss_fn(args, ptoSolver, conf):
    name = args.opt_model
    if name == "mse":
        from openpto.method.Models.MSE import MSE

        ModelCalss = MSE
    elif name == "msesum":
        from openpto.method.Models.MSE import MSE_Sum

        ModelCalss = MSE_Sum
    elif name == "ce":
        from openpto.method.Models.MSE import CE

        ModelCalss = CE
    elif name == "bce":
        from openpto.method.Models.MSE import BCE

        ModelCalss = BCE
    elif name == "mae":
        from openpto.method.Models.MSE import MAE

        ModelCalss = MAE
    return ModelCalss


def get_loss_fn(args, ptoSolver, conf):
    name = args.opt_model
    if name in ["mse", "msesum", "ce", "bce", "mae"]:
        ModelCalss = get_ml_loss_fn(args, ptoSolver, conf)
    elif name == "dfl":
        from openpto.method.Models.MSE import DFL

        ModelCalss = DFL
    elif name == "spo":
        from openpto.method.Models.SPO import SPO

        ModelCalss = SPO
    elif name == "pointLTR":
        from openpto.method.Models.LTR import pointwiseLTR

        ModelCalss = pointwiseLTR
    elif name == "pairLTR":
        from openpto.method.Models.LTR import pairwiseLTR

        ModelCalss = pairwiseLTR
    elif name == "listLTR":
        from openpto.method.Models.LTR import listwiseLTR

        ModelCalss = listwiseLTR
    elif name == "qptl":
        from openpto.method.Models.QPTL import QPTL

        ModelCalss = QPTL
    elif name == "intopt":
        # from openpto.method.Models.Intopt import Intopt
        ModelCalss = None
    elif name == "nce":
        from openpto.method.Models.NCE import NCE

        ModelCalss = NCE
    elif name == "blackboxSolver":
        from openpto.method.Models.Blackbox import blackboxSolver

        ModelCalss = blackboxSolver
    elif name == "blackbox":
        from openpto.method.Models.Blackbox import subopt_blackbox

        ModelCalss = subopt_blackbox
    elif name == "identitySolver":
        from openpto.method.Models.Identity import IdentitySolver

        ModelCalss = IdentitySolver
    elif name == "identity":
        from openpto.method.Models.Identity import subopt_Identity

        ModelCalss = subopt_Identity
    elif name == "lodl":
        from openpto.method.Models.LODLs import LODL

        ModelCalss = LODL
    elif name == "perturb":
        from openpto.method.Models.perturbed import perturbed

        ModelCalss = perturbed
    elif name == "cpLayer":
        from openpto.method.Models.cpLayer import cpLayer

        ModelCalss = cpLayer
    else:
        raise LookupError()
    loss_dict = {
        **conf["models"][args.opt_model],
        "log_dir": args.log_dir,
        "loss_path": args.loss_path,
    }
    return ModelCalss(ptoSolver, **loss_dict)
