import argparse
import ast
import os
import random

import numpy as np
import ruamel.yaml as yaml
import torch

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
    parser.add_argument("--config_path", type=str, default="")
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
            "mae",
            "spo",
            "pointLTR",
            "pairLTR",
            "listLTR",
            "intopt",
            "blackbox",
            "identity",
            "lodl",
            "nce",
            "qptl",
            "lodl",
        ],
        default="mse",
    )
    parser.add_argument(
        "--pred_model", type=str, choices=["LR", "dense"], default="dense"
    )
    parser.add_argument(
        "--solver", type=str, choices=["gurobi", "neural", "heuristic", "cvxpy"], default="gurobi"
    )
    parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")

    # training
    parser.add_argument("--loadnew", type=ast.literal_eval, default=False)
    parser.add_argument("--n_epochs", type=int, default=0)
    parser.add_argument("--n_ptr_epochs", type=int, default=0)
    parser.add_argument("--earlystopping", type=ast.literal_eval, default=True)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--pred_bz", type=int, default=8192)
    # data
    parser.add_argument("--data_dir", type=str, default="./openpto/data/")
    parser.add_argument("--do_debug", action="store_true")
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--testinstances", type=int, default=200)
    # debug
    # parser.add_argument("--valfrac", type=float, default=0.5)
    parser.add_argument("--valfreq", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="default")
    # model
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=32)
    # solver
    parser.add_argument("--solve_ratio", type=float, default=1.0)
    parser.add_argument("--processes", type=int, default=1)
    #   Decision-Focused Learning
    # parser.add_argument("--numsamples", type=int, default=5000)
    parser.add_argument("--losslr", type=float, default=0.01)

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
    if path == "":
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


def get_logger(logger_fname):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s:  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(f"{logger_fname}/log.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


################################### Seed ###################################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
