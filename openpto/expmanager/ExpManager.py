import torch
# from openpto.utils.utils import set_seed
# from openpto.utils.logger import Logger
# from openpto.config.util import save_conf
import os
import time as time


class ExpManager:
    '''
    Experiment management class to enable running multiple experiment,
    loading learned structures and saving results.

    Parameters
    ----------
    solver : openpto.method.Solver
        Solver of the method to solve the task.
    n_splits : int
        Number of data splits to run experiment on.
    n_runs : int
        Number of experiment runs each split.
    save_path : str
        Path to save the config file.
    debug : bool
        Whether to print statistics during training.

    Examples
    --------
    >>> # load dataset
    >>> import openpto.dataset
    >>> dataset = openpto.dataset.Dataset('cora', feat_norm=True)
    >>> # load config file
    >>> import openpto.config.load_conf
    >>> conf = openpto.config.load_conf('gcn', 'cora')
    >>> # create solver
    >>> import openpto.method.SGCSolver
    >>> solver = SGCSolver(conf, dataset)
    >>>
    >>> import openpto.ExpManager
    >>> exp = ExpManager(solver)
    >>> exp.run(n_runs=10, debug=True)

    '''
    def __init__(self, solver=None, save_path=None):
        self.solver = solver
        self.conf = solver.conf
        self.method = solver.method_name
        self.dataset = solver.dataset
        self.data = self.dataset.name
        self.device = torch.device('cuda')
        # you can change random seed here
        self.train_seeds = [i for i in range(400)]
        self.split_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.save_path = None
        self.save_graph_path = None
        self.load_graph_path = None
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_path = save_path
        # if 'save_graph' in self.conf.analysis and self.conf.analysis['save_graph']:
        #     assert 'save_graph_path' in self.conf.analysis and self.conf.analysis['save_graph_path'] is not None, 'Specify the path to save graph'
        #     self.save_graph_path = os.path.join(self.conf.analysis['save_graph_path'], self.method)
        # if 'load_graph' in self.conf.analysis and self.conf.analysis['load_graph']:
        #     assert 'load_graph_path' in self.conf.analysis and self.conf.analysis[
        #         'load_graph_path'] is not None, 'Specify the path to load graph'
        #     self.load_graph_path = self.conf.analysis['load_graph_path']
        assert self.save_graph_path is None or self.load_graph_path is None, 'GNN does not save graph, GSL does not load graph'

    def run(self, n_splits=1, n_runs=1, debug=False):
        retrun