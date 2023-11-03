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
            "portfolio",
            "knapsack",
            "energy",
            "advertising",
        ],
        default="knapsack",
    )
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument(
        "--method_path", type=str, default="openpto/config/models/default.yaml"
    )
    parser.add_argument("--trained_path", type=str, default="")
    parser.add_argument("--loss_path", type=str, default="")
    parser.add_argument(
        "--opt_model",
        type=str,
        choices=[
            "mse",
            "dfl",
            "bce",
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
            "perturb",
            "cpLayer",
        ],
        default="mse",
    )
    parser.add_argument(
        "--pred_model", type=str, choices=["dense", "cvr"], default="dense"
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=[
            "gurobi",
            "neural",
            "heuristic",
            "cvxpy",
            "ortools",
            "qptl",
        ],
        default="gurobi",
    )
    parser.add_argument("--gpu", type=str, default="-1", help="Visible GPU")

    # training
    parser.add_argument("--loadnew", type=ast.literal_eval, default=False)
    parser.add_argument("--n_epochs", type=int, default=0)
    parser.add_argument("--n_ptr_epochs", type=int, default=0)
    parser.add_argument("--earlystopping", type=ast.literal_eval, default=True)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batchsize", type=int, default=1)
    # data
    parser.add_argument("--data_dir", type=str, default="./openpto/data/")
    parser.add_argument("--do_debug", action="store_true")
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--testinstances", type=int, default=200)
    # debug
    parser.add_argument("--valfreq", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="default")
    # model
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=32)
    # solver
    parser.add_argument("--solve_ratio", type=float, default=1.0)
    parser.add_argument("--processes", type=int, default=1)
    #
    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.problem)
    return args


###################################### Configs ###############################################


def load_conf(prob_path: str = None, method_path: str = None, prob_name: str = None):
    """
    Function to load config file.

    Parameters
    ----------
    prob_path : str
        Path to load config file. Load default configuration if set to `None`.
    method_path : str
        Path to load method config file. Necessary if ``path`` is set to `None`.
    prob_name : str
        Name of the corresponding problem. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    """
    if prob_path == "":
        dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/probs/"
        )
        prob_path = os.path.join(dir, prob_name + ".yaml")

    if os.path.exists(prob_path) is False:
        raise ValueError(f"The configuration file, [{prob_path}] is not provided.")

    conf = yaml.safe_load(open(prob_path, "r").read())
    conf["models"] = yaml.safe_load(open(method_path, "r").read())
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
