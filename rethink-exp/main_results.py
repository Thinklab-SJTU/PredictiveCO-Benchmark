import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='KP',
                    choices=['KP','Energy'], help='dataset')
parser.add_argument('--method', type=str, default='gcn', choices=['QPTL'], help="Select methods")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from openpto.config import load_conf
from openpto.data import Dataset
from openpto import ExpManager
from openpto.method import *

conf = load_conf(method=args.method, dataset=args.data)
# if not args.method in ['gcn']:
#     conf.analysis['save_graph'] = True
#     conf.analysis['save_graph_path'] = 'results/graph'
# else:
#     conf.analysis['save_graph'] = False
print(conf)

dataset = Dataset(args.data, feat_norm=conf.dataset['feat_norm'], path='data')


method = eval('{}Solver(conf, dataset)'.format(args.method.upper()))
exp = ExpManager(method, n_runs=1, debug=args.debug, save_path='records')
exp.run()