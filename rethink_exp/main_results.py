import os
import sys


# Makes sure hashes are consistent
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from openpto import ExpManager
from openpto.method.models.loss import get_loss_fn
from openpto.utils import get_args
from openpto.config import load_conf
from openpto.metrics import *
from openpto.problems.wrapper_prob import problem_wrapper
from openpto.method import *

if __name__ == '__main__':
    args = get_args()
    print(f"Hyperparameters: {args}\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load problem
    conf = load_conf(method_name=args.opt_model, prob_name=args.problem)
    print(f"Loading [{args.problem}] Problem... Config: {conf}")
    problem = problem_wrapper(args, conf)

    # Load loss function
    print(f"Loading [{args.opt_model}] Loss Function...")
    loss_fn = get_loss_fn(
        args.opt_model,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        rank=args.quadrank,
        sampling_std=args.samplingstd,
        quadalpha=args.quadalpha,
        lr=args.lr,
        serial=args.serial,
        dflalpha=args.dflalpha,
    )

    ipdim, opdim = problem.get_model_shape()
    model_args = {"ipdim":ipdim, "opdim":opdim, "out_act": problem.get_output_activation()}
    exp = ExpManager(model_args, save_path='saved_records', args = args)

    # Train neural network with a given loss function
    print(f"Start training [{args.pred_model}] model on [{args.opt_model}] loss...")
    exp.run(problem, loss_fn, n_epochs=args.epochs)