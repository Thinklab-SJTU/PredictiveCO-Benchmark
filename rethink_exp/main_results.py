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

# from models import model_dict
# from losses import MSE, get_loss_fn
# from utils import print_metrics, init_if_not_saved, move_to_gpu
from openpto.utils.utils import get_args,print_metrics, init_if_not_saved, move_to_gpu




# from openpto.config import load_conf
from openpto import ExpManager
from openpto.problems.prob_utils import str2prob
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


    method = eval('{}Solver(conf, problem)'.format(args.method.upper()))
    exp = ExpManager(method, n_runs=1, debug=args.debug, save_path='saved_records')
    exp.run()