import os

NUM_CPUS = os.cpu_count()


def get_loss_fn(name, problem, **kwargs):
    if name == "mse":
        from openpto.method.Models.MSE import MSE

        return MSE
    elif name == "msesum":
        from openpto.method.Models.MSE import MSE_Sum

        return MSE_Sum
    elif name == "ce":
        from openpto.method.Models.MSE import CE

        return CE
    elif name == "bce":
        from openpto.method.Models.MSE import BCE

        return BCE
    elif name == "mae":
        from openpto.method.Models.MSE import MAE

        return MAE
    elif name == "dfl":
        from openpto.method.Models.MSE import DFL

        return DFL
    elif name == "spo":
        from openpto.method.Models.SPO import SPO

        return SPO
    elif name == "pointLTR":
        from openpto.method.Models.LTR import pointwiseLTR

        return pointwiseLTR
    elif name == "pairLTR":
        from openpto.method.Models.LTR import pairwiseLTR

        return pairwiseLTR
    elif name == "listLTR":
        from openpto.method.Models.LTR import listwiseLTR

        return listwiseLTR
    elif name == "qptl":
        # from openpto.method.Models.QPTL import QPTL
        # return QPTL
        return None
    elif name == "intopt":
        # from openpto.method.Models.Intopt import Intopt
        return None
    elif name == "nce":
        from openpto.method.Models.NCE import NCE

        return NCE
    elif name == "blackbox":
        from openpto.method.Models.Blackbox import blackbox

        return blackbox
    elif name == "identity":
        from openpto.method.Models.Identity import negativeIdentity

        return negativeIdentity
    elif name == "lodl":
        from openpto.method.Models.LODLs import LODL

        return LODL
    elif name == "perturb":
        from openpto.method.Models.perturbed import perturbed

        return perturbed
    else:
        raise LookupError()
