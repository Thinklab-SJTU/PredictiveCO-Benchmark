from itertools import repeat

import pandas as pd
import numpy as np

import torch

def starmap_with_kwargs(pool, fn, args_iter, kwargs):
    args_for_starmap = zip(repeat(fn), args_iter, repeat(kwargs))
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def gather_incomplete_left(tensor, I):
    return tensor.gather(I.ndim, I[(...,) + (None,) * (tensor.ndim - I.ndim)].expand((-1,) * (I.ndim + 1) + tensor.shape[I.ndim + 1:])).squeeze(I.ndim)

def trim_left(tensor):
    while tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor

class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.shape[:-1]
        shape = (*batch_size, *self.shape)
        out = input.view(shape)
        return out

def solve_lineqn(A, b, eps=1e-5):
    try:
        result = torch.linalg.solve(A, b)
    except RuntimeError:
        print(f"WARNING: The matrix was singular")
        result = torch.linalg.solve(A + eps * torch.eye(A.shape[-1]), b)
    return result

############################### Solve ##################################
def GrbSolve(cp, optmodel):
    ins_num = len(cp) if cp.ndim == 2 else 1
    sol = []
    obj = []
    for i in range(ins_num):
        # solve
        optmodel.setObj(cp[i])
        solp, objp = optmodel.solve()
        sol.append(solp)
        obj.append(objp)
    return sol, obj

def _solve_in_pass(cp, optmodel, processes, pool):
    """
    A function to solve optimization in the forward/backward pass
    """
    # number of instance

    # single-core
    if processes == 1:
        sol, obj = GrbSolve(cp, optmodel)
    # multi-core
    else:
        raise NotImplementedError("Parallel computing is not supported yet.")
        # # get class
        # model_type = type(optmodel)
        # # get args
        # args = getArgs(optmodel)
        # # parallel computing
        # res = pool.amap(_solveWithObj4Par, cp, [args] * ins_num,
        #                 [model_type] * ins_num).get()
        # # get res
        # sol = np.array(list(map(lambda x: x[0], res)))
        # obj = np.array(list(map(lambda x: x[1], res)))
    return sol, obj
