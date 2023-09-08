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
    elif name == "dfl":
        from openpto.method.Models.MSE import DFL

        return DFL
    elif name == "spo":
        from openpto.method.Models.SPO import SPOPlus

        return SPOPlus
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
        return None
        # return QPTL
    elif name == "intopt":
        # from openpto.method.Models.Intopt import Intopt
        return None
    elif name == "nce":
        from openpto.method.Models.NCE import NCE

        return NCE
    elif name == "blackbox":
        from openpto.method.Models.Blackbox import blackboxOpt

        return blackboxOpt
    elif name == "identity":
        from openpto.method.Models.Identity import negativeIdentity

        return negativeIdentity
    elif name == "lodl":
        return None
        # return _get_learned_loss(problem, name, **kwargs)

    else:
        raise LookupError()
