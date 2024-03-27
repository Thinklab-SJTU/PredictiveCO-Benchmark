import os
import sys
import warnings

import torch

from openpto.config import get_args, get_logger, load_conf, setup_seed
from openpto.expmanager import ExpManager
from openpto.method.Models.wrapper_loss import get_loss_fn
from openpto.method.Solvers.wrapper_solver import solver_wrapper
from openpto.problems.wrapper_prob import problem_wrapper

warnings.filterwarnings("ignore")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Makes sure hashes are consistent
hashseed = os.getenv("PYTHONHASHSEED")
if not hashseed:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == "__main__":
    # get configs
    args = get_args()
    conf = load_conf(args.config_path, args.method_path, args.problem)

    # set seed
    setup_seed(args.seed)

    # set logger
    logger = get_logger(args, conf)
    logger.info(f" {args.bkup_log_dir}\n {args.log_dir}\n args: {args} \n")

    # Load problem
    logger.info(f" dataset configs: {conf['dataset']} \n")
    logger.info(f" model configs: {conf['models'][args.opt_model]} \n")
    logger.info(f" Loading [{args.problem}] Problem...")
    problem = problem_wrapper(args, conf)

    # Load solver
    logger.info(f" Loading [{args.solver}] solver ...")
    ptoSolver = solver_wrapper(args, conf, problem)

    # Load loss function
    logger.info(f" Loading [{args.opt_model}] Loss Function...")
    loss_fn = get_loss_fn(args, ptoSolver, conf)

    # load exp manager
    pred_model_args = {
        "ipdim": problem.get_model_shape()[0],
        "opdim": problem.get_model_shape()[1],
        "out_act": problem.get_output_activation(),
    }
    exp = ExpManager(pred_model_args, args=args, conf=conf, logger=logger)

    # Train neural network with a given loss function
    logger.info(
        f" Start training [{args.pred_model}] model on [{args.opt_model}] loss..."
    )
    exp.run(problem, loss_fn, ptoSolver, n_epochs=args.n_epochs, do_debug=args.do_debug)
