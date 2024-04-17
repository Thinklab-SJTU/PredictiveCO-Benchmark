#!/usr/bin/env python
# coding: utf-8
"""

"""

import heapq
import itertools

from functools import partial

import numpy as np

from openpto.method.Solvers.abcptoSolver import ptoSolver


class spSolver(ptoSolver):
    """ """

    def __init__(self, modelSense, n_vars, size, neighbourhood_fn, **kwargs):
        super().__init__(modelSense)
        self.n_vars = n_vars
        self.size = size
        self.neighbourhood_fn = neighbourhood_fn

    def solve(self, matrix, do_debug=False, **kwargs):
        """
        dijkstra solver
        """
        matrix = matrix.reshape(self.size, self.size)
        x_max, y_max = matrix.shape
        neighbors_func = partial(
            get_neighbourhood_func(self.neighbourhood_fn), x_max=x_max, y_max=y_max
        )
        #
        costs = np.full_like(matrix, 1.0e10)
        costs[0][0] = matrix[0][0]
        # if do_debug:
        #     print("initial: ", costs[0][0])
        num_path = np.zeros_like(matrix)
        num_path[0][0] = 1
        priority_queue = [(matrix[0][0], (0, 0))]
        certain = set()
        transitions = dict()

        while priority_queue:
            cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
            if (cur_x, cur_y) in certain:
                pass
            for x, y in neighbors_func(cur_x, cur_y):
                # if do_debug:
                #     print("x,y:", x, y, end="  ")
                if (x, y) not in certain:
                    # if do_debug:
                    #     print(
                    #         "  compare: ", matrix[x][y] + costs[cur_x][cur_y], costs[x][y]
                    #     )
                    if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                        costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                        heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                        transitions[(x, y)] = (cur_x, cur_y)
                        num_path[x, y] = num_path[cur_x, cur_y]
                    elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                        num_path[x, y] += 1

            certain.add((cur_x, cur_y))
        # retrieve the path
        cur_x, cur_y = x_max - 1, y_max - 1
        on_path = np.zeros_like(matrix)
        on_path[-1][-1] = 1
        # if do_debug:   print("transitions: ", transitions.keys())
        while (cur_x, cur_y) != (0, 0):
            # if (cur_x, cur_y) not in transitions.keys():
            #     return np.zeros(self.size * self.size), {}
            # print("exists: ", (cur_x, cur_y) in transitions.keys(), end=" ")
            cur_x, cur_y = transitions[(cur_x, cur_y)]
            on_path[cur_x, cur_y] = 1.0

        is_unique = num_path[-1, -1] == 1

        Z = on_path.reshape(-1)
        others = {"is_unique": is_unique, "transitions": transitions}
        return Z, others
        # return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)


def neighbours_8(x, y, x_max, y_max):
    deltas_x = (-1, 0, 1)
    deltas_y = (-1, 0, 1)
    for dx, dy in itertools.product(deltas_x, deltas_y):
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def neighbours_4(x, y, x_max, y_max):
    for dx, dy in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def get_neighbourhood_func(neighbourhood_fn):
    if neighbourhood_fn == "4-grid":
        return neighbours_4
    elif neighbourhood_fn == "8-grid":
        return neighbours_8
    else:
        raise Exception(f"neighbourhood_fn of {neighbourhood_fn} not possible")
