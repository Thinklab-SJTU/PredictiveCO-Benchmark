from functools import partial
import os
import sys
from copy import deepcopy
import random
import pdb

# Makes sure hashes are consistent
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)


import matplotlib.pyplot as plt

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


from openpto import ExpManager
from openpto.method.loss_utils import get_loss_fn
from openpto.utils import get_args
# from openpto.config import load_conf
from openpto.problems.prob_utils import str2prob, init_if_not_saved
from openpto.method import *

if __name__ == '__main__':
    args = get_args()
    # conf = load_conf(method=args.method, dataset=args.data)
    # print(conf)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")
    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
    ProblemClass = str2prob(args.problem)
    # problem_kwargs = {}
    # TODO: automate kwargs
    problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_targets': args.numtargets,
                            'num_items': args.numitems,
                            'budget': args.budget,
                            'num_fake_targets': args.fakefeatures,
                            'rand_seed': args.seed,
                            'val_frac': args.valfrac,}
    problem = init_problem(ProblemClass, problem_kwargs)
    # dataset = Dataset(args.data, feat_norm=conf.dataset['feat_norm'], path='data')

    
    # Load an ML model to predict the parameters of the problem
    ipdim, opdim = problem.get_modelio_shape()
    prob_args = {"ipdim":ipdim, "opdim":opdim, "out_act": problem.get_output_activation()}


    print(f"Loading {args.loss} Loss Function...")
    loss_fn = get_loss_fn(
        args.loss,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        rank=args.quadrank,
        sampling_std=args.samplingstd,
        quadalpha=args.quadalpha,
        lr=args.losslr,
        serial=args.serial,
        dflalpha=args.dflalpha,
    )

    # Train neural network with a given loss function
    print(f"Start training [{args.model}] model on [{args.loss}] loss...")
    
    # method = eval('{}Solver(conf, problem)'.format(args.method.upper()))
    method = None
    exp = ExpManager(prob_args, save_path='saved_records', args = args)
    exp.run(problem, loss_fn, n_epochs=args.epochs)