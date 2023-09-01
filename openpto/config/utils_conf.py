import argparse
import ast
import os

import ruamel.yaml as yaml

###################################### Args ###############################################


def get_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument(
        "--problem",
        type=str,
        choices=[
            "budgetalloc",
            "bipartitematching",
            "cubic",
            "rmab",
            "portfolio",
            "knapsack",
            "energy",
        ],
        default="portfolio",
    )
    parser.add_argument("--prob_version", type=str, default=None)
    parser.add_argument(
        "--opt_model",
        type=str,
        choices=[
            "mse",
            "msesum",
            "dense",
            "weightedmse",
            "weightedmse++",
            "weightedce",
            "weightedmsesum",
            "dfl",
            "quad",
            "quad++",
            "ce",
            "spo",
            "ltr",
            "intopt",
            "blackbox",
            "Identity",
            "LODL",
        ],
        default="mse",
    )
    parser.add_argument(
        "--pred_model", type=str, choices=["LR", "dense"], default="dense"
    )
    parser.add_argument("--solver", type=str, choices=["gurobi","neural"], default="neural")
    parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")

    # training
    parser.add_argument("--loadnew", type=ast.literal_eval, default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--earlystopping", type=ast.literal_eval, default=True)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=1000)
    # data
    parser.add_argument("--data_dir", type=str, default="./openpto/data/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--testinstances", type=int, default=200)
    # debug
    parser.add_argument("--valfrac", type=float, default=0.5)
    parser.add_argument("--valfreq", type=int, default=5)

    # model
    parser.add_argument("--layers", type=int, default=2)
    #   Decision-Focused Learning
    parser.add_argument("--dflalpha", type=float, default=1.0)
    #   Learned-Loss
    parser.add_argument("--serial", type=ast.literal_eval, default=True)
    parser.add_argument(
        "--sampling",
        type=str,
        choices=[
            "random",
            "random_flip",
            "random_uniform",
            "numerical_jacobian",
            "random_jacobian",
            "random_hessian",
            "random",
        ],
        default="random",
    )
    parser.add_argument("--samplingstd", type=float)
    parser.add_argument("--numsamples", type=int, default=5000)
    parser.add_argument("--losslr", type=float, default=0.01)
    #       Approach-Specific: Quadratic
    parser.add_argument("--quadrank", type=int, default=20)
    parser.add_argument("--quadalpha", type=float, default=0)
    args = parser.parse_args()
    return args


###################################### Configs ###############################################


def load_conf(path: str = None, method_name: str = None, prob_name: str = None):
    """
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    """
    if path is None and method_name is None:
        raise KeyError
    if path is None and prob_name is None:
        raise KeyError
    if path is None:
        # method_names = ['spo','ltr','intopt','nce','blackbox']
        # prob_names = ['knapsack', '']

        # assert method in method_name
        # assert prob_name in prob_names
        dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/probs/"
        )
        path = os.path.join(dir, prob_name + ".yaml")

        if os.path.exists(path) is False:
            raise KeyError("The configuration file is not provided.")

    conf = open(path, "r").read()
    conf = yaml.safe_load(conf)
    # conf = argparse.Namespace(**conf)

    return conf


def save_conf(path, conf):
    """
    Function to save the config file.

    Parameters
    ----------
    path : str
        Path to save config file.
    conf : dict
        The config dict.

    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(vars(conf), f)
