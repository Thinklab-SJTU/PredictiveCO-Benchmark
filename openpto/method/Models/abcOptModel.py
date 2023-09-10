from abc import abstractmethod

from torch import nn


class optModel(nn.Module):
    """
    An abstract module for the learning to rank losses, which measure the difference in how the predicted cost
    vector and the true cost vector rank a pool of feasible solutions.
    """

    def __init__(self, optSolver=None, processes=1, solve_ratio=1, **kwargs):
        super(optModel, self).__init__()
        self.optSolver = optSolver
        self.solve_ratio = solve_ratio
        # TODO: multi-process
        # number of processes
        # if processes not in range(mp.cpu_count() + 1):
        #     raise ValueError(
        #         "Invalid processors number {}, only {} cores.".format(
        #             processes, mp.cpu_count()
        #         )
        #     )
        # self.processes = mp.cpu_count() if not processes else processes
        self.processes = processes
        # # single-core
        if processes == 1:
            self.pool = None
        # multi-core
        else:
            assert 0
            # self.pool = ProcessingPool(processes)
        # print("Num of cores: {}".format(self.processes))

    @abstractmethod
    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true=None,
        params=None,
        **hyperparams,
    ):
        """
        Input:
            problem:
            coeff_hat:
            coeff_true:
            params:
        Output:
            sol, obj, loss
        """

        pass
