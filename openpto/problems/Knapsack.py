import os
import itertools
import random

# import pandas as pd
import numpy as np

import torch

from openpto.problems.PTOProblem import PTOProblem
from openpto.problems.utils_prob import read_file, generate_uniform_weights_from_seed

BENCHMARK_SIZE = 48

class Knapsack(PTOProblem):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_items=100,  # number of targets to consider
        num_fake_targets=5000,  # number of random features added to make the task harder
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(Knapsack, self).__init__()
        self._set_seed(rand_seed)
        generate_weight = True
        unit_weight = False
        kfold = [0,1,2,3,4]
        noise_level = 0
        # TODO:automate
        dataset = get_energy_data('energy_data.txt', generate_weight=generate_weight, unit_weight=unit_weight,
                                  kfold=kfold, noise_level=noise_level)
    
    def get_train_data(self, **kwargs):
        raise NotImplementedError()

    def get_val_data(self, **kwargs):
        raise NotImplementedError()

    def get_test_data(self, **kwargs):
        raise NotImplementedError()
    
    def get_model_shape(self):
        raise NotImplementedError()

    def get_output_activation(self):
        raise NotImplementedError()

    @staticmethod
    def genData(num_data, num_features, num_items, dim=1, deg=1, noise_width=0, seed=135):
    #     A function to generate synthetic data and features for knapsack

    #     Args:
    #         num_data (int): number of data points
    #         num_features (int): dimension of features
    #         num_items (int): number of items
    #         dim (int): dimension of multi-dimensional knapsack
    #         deg (int): data polynomial degree
    #         noise_width (float): half witdth of data random noise
    #         seed (int): random state seed

    #     Returns:
    #     tuple: weights of items (np.ndarray), data features (np.ndarray), costs (np.ndarray)
        # positive integer parameter
        if type(deg) is not int:
            raise ValueError("deg = {} should be int.".format(deg))
        if deg <= 0:
            raise ValueError("deg = {} should be positive.".format(deg))
        # set seed
        rnd = np.random.RandomState(seed)
        # number of data points
        n = num_data
        # dimension of features
        p = num_features
        # dimension of problem
        d = dim
        # number of items
        m = num_items
        # weights of items
        weights = rnd.choice(range(300, 800), size=(d,m)) / 100
        # random matrix parameter B
        B = rnd.binomial(1, 0.5, (m, p))
        # feature vectors
        x = rnd.normal(0, 1, (n, p))
        # value of items
        c = np.zeros((n, m), dtype=int)
        for i in range(n):
            # cost without noise
            values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
            # rescale
            values *= 5
            values /= 3.5 ** deg
            # noise
            epislon = rnd.uniform(1 - noise_width, 1 + noise_width, m)
            values *= epislon
            # convert into int
            values = np.ceil(values)
            c[i, :] = values
            # float
            c = c.astype(np.float64)
        return weights, x, c

    
def get_energy_data(filename, generate_weight = True, unit_weight = True, kfold=0, noise_level = 0, is_spo_tree=False):
    """
    Reads the energy dataset with the filename, splits it into feature and output sets.
    :param filename:
    :return:dataset: contains X and Y(nparray), dataset_params: contains feature_size and sample_size (int)
    """
    HEADER_LENGTH = 4
    dir_path = os.getcwd()
    # dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(dir_path, 'data', filename)
    data = read_file(file_path)
    dataset = transform_energy_data(data, HEADER_LENGTH, generate_weight, unit_weight, kfold,noise_level=noise_level, is_spo_tree=is_spo_tree)
    return dataset

def transform_energy_data(data, header_length, generate_weight=True, unit_weight = True, kfold=0,noise_level=0,is_spo_tree=False):
    """
    transform method for energy data. Takes raw file and splits it into features and labels.
    For the energy data, first feature is actually the benchmark No
    :param data (list): List contains the data set:
    :param header_length (int): Length of header in the file. Used to decide where the itemset starts
    :return: Dataset(dictionary) holds two nparry, X: features , Y: labels. dataset_params(dictionary) contains feature_size and sample_size
    """
    weight_seed = np.array([3,5,7])
    # weights = np.random.choice(weight_seed, Y.size)
    # X = np.vstack([X, weights])
    # Y = Y * weights

    sample_size = int(data[header_length][0])
    feature_size = int(data[header_length][1])
    data = np.array(data[(header_length + 1):])

    X = data[:, 0:feature_size]
    X = np.asfarray(np.array(X), float).T # (9, 37872)
    X = X.reshape(feature_size, sample_size)

    Y = data[:, feature_size]
    Y = np.asfarray(np.array(Y), float).T
    Y = Y.reshape(1, sample_size)

    k_fold_rotation = int(48 * (750 / 5))
    # print("X shape: ",X.shape,kfold, k_fold_rotation, kfold*k_fold_rotation)
    X = np.roll(X, kfold*k_fold_rotation, axis=1)
    Y = np.roll(Y, kfold*k_fold_rotation, axis=1)
    # X = X[[0,7],:].reshape(2,-1)
    if is_spo_tree:
        dataset = get_benchmarks_spotree(X,Y, BENCHMARK_SIZE, generate_weight, unit_weight, weight_seed,noise_level=noise_level)
    else:
        dataset = get_benchmarks(X,Y, BENCHMARK_SIZE, generate_weight, unit_weight, weight_seed,noise_level=noise_level)

    # dataset = {'X': X,
    #            'Y': Y}
    #
    dataset['feature_size'] = feature_size - 1,
    dataset['sample_size'] = sample_size
    dataset['benchmark_size'] = BENCHMARK_SIZE
    return dataset

def get_benchmarks(X, Y, benchmark_size, generate_weight=True, unit_weight=True, weight_seed=None, add_weights=False,noise_level=0):
    """
    Splits the dataset into benchmarks of a certain size for the optimization problem. Used in the second stage.
    Might not be used in the feature if we choose to use predetermined benchmarks.
    :param is_weighted:
    :param X:
    :param Y:
    :param benchmark_size:
    :return:
    """

    #Add time slot number
    # time_slots = np.array(list(np.array([x for x in range(48)])) * int(Y.shape[1]/48))
    # time_slots_rev = np.array(list(np.array([x for x in reversed(range(48))])) * int(Y.shape[1] / 48))
    # time_slots = one_hot_timeslot(int(Y.shape[1]/48))
    # X = np.vstack((X, time_slots))
    # X = np.vstack((X, time_slots_rev.reshape(1, -1)))

    feature_size, sample_size = X.shape
    if weight_seed is not None:
        if unit_weight:
            weight_seed = [1]
        else:
            weight_seed = [3, 5, 7]
            # IMPLEMENT ADDING NOISE
            np.random.seed(RANDOM_SEED)
            # if noise_level > 0:
            #     print('noise generate')
            #     noise = (1-np.random.random(sample_size)*noise_level/100)
            #     Y = Y * noise

    benchmark_count = int(sample_size / benchmark_size)
    benchmarks_X = []
    benchmarks_Y = []
    benchmarks_weights = []

    # do weights if needed
    if generate_weight:
        weights = np.ones(Y.shape)
        noisy_weights =  np.ones(Y.shape)
    else:
        weights = X[-1, :].reshape(Y.shape)
    for i in range(benchmark_count):
        start_index = i * benchmark_size
        end_index = start_index + benchmark_size

        benchmark_X = X[1:, start_index:end_index].reshape(feature_size - 1, benchmark_size)

        if generate_weight:
            if unit_weight:
                benchmark_weights = generate_uniform_weights_from_seed(benchmark_size, weight_seed)
                benchmark_noisy_weights = benchmark_weights
            else:
                #set same weight array for SPO implementation
                benchmark_weights = np.array(
                    [5, 3, 3, 5, 5, 7, 7, 3, 7, 7, 3, 3, 5, 3, 7, 3, 7, 7, 5, 5, 3, 5, 5, 3, 7, 7, 3, 7, 5, 5, 7, 3, 7,
                     3,
                     3, 5, 7, 5, 3, 5, 3, 7, 5, 7, 5, 5, 3, 7]).reshape(1, 48)
                benchmark_noisy_weights = benchmark_weights+(np.ones(benchmark_weights.shape)*noise_level)
            # benchmark_weights = np.hstack([seed_array for i in range(int(sample_size/len(seed_array)))])
            Y[:, start_index:end_index] = Y[:, start_index:end_index] * benchmark_noisy_weights
            weights[:, start_index:end_index] = benchmark_weights
            noisy_weights[:, start_index:end_index] = benchmark_noisy_weights
            benchmark_X = np.vstack((benchmark_X, benchmark_noisy_weights   .flatten()))

        else:
            benchmark_weights = weights[:, start_index:end_index].reshape(1, benchmark_size)

        benchmark_Y = Y[:, start_index:end_index].reshape(1, benchmark_size)

        benchmarks_X.append(benchmark_X)
        benchmarks_Y.append(benchmark_Y)
        benchmarks_weights.append(benchmark_weights)

    if generate_weight and not unit_weight and add_weights:
        X = np.vstack((X, benchmark_noisy_weights.flatten()))

    X = np.delete(X, 0, 0)

    dataset = {'X': X,
               'Y': Y,
               'weights': weights,
               'benchmarks_X': benchmarks_X,
               'benchmarks_Y': benchmarks_Y,
               'benchmarks_weights': benchmarks_weights}
    return dataset

def get_benchmarks_spotree(X, Y, benchmark_size, generate_weight=True, unit_weight=True, weight_seed=None, add_weights=True,noise_level=0):
    """
    Splits the dataset into benchmarks of a certain size for the optimization problem. Used in the second stage.
    Might not be used in the feature if we choose to use predetermined benchmarks.
    :param is_weighted:
    :param X:
    :param Y:
    :param benchmark_size:
    :return:
    """

    feature_size, sample_size = X.shape
    if weight_seed is not None:
        if unit_weight:
            weight_seed = [1]
        else:
            weight_seed = [3, 5, 7]
            # IMPLEMENT ADDING NOISE
            np.random.seed(RANDOM_SEED)
            # if noise_level > 0:
            #     print('noise generate')
            #     noise = (1-np.random.random(sample_size)*noise_level/100)
            #     Y = Y * noise

    benchmark_count = int(sample_size / benchmark_size)
    benchmarks_X = []
    benchmarks_Y = []
    benchmarks_weights = []

    # do weights if needed
    if generate_weight:
        weights = np.ones(Y.shape)
        noisy_weights =  np.ones(Y.shape)
    else:
        weights = X[-1, :].reshape(Y.shape)
    for i in range(benchmark_count):
        start_index = i * benchmark_size
        end_index = start_index + benchmark_size

        benchmark_X = X[1:, start_index:end_index].reshape(feature_size - 1, benchmark_size)
        if generate_weight:
            if unit_weight:
                benchmark_weights = generate_uniform_weights_from_seed(benchmark_size, weight_seed).astype(int)
                benchmark_noisy_weights = benchmark_weights
            else:
                #set same weight array for SPO implementation
                benchmark_weights = np.array(
                    [5, 3, 3, 5, 5, 7, 7, 3, 7, 7, 3, 3, 5, 3, 7, 3, 7, 7, 5, 5, 3, 5, 5, 3, 7, 7, 3, 7, 5, 5, 7, 3, 7,
                     3,
                     3, 5, 7, 5, 3, 5, 3, 7, 5, 7, 5, 5, 3, 7]).reshape(1, 48)
                benchmark_noisy_weights = benchmark_weights+(np.ones(benchmark_weights.shape)*noise_level)
            # benchmark_weights = np.hstack([seed_array for i in range(int(sample_size/len(seed_array)))])
            Y[:, start_index:end_index] = Y[:, start_index:end_index] * benchmark_noisy_weights
            weights[:, start_index:end_index] = benchmark_weights
            noisy_weights[:, start_index:end_index] = benchmark_noisy_weights
            benchmark_X = np.vstack((benchmark_X, benchmark_noisy_weights   .flatten()))
            # benchmark_X = np.mean(benchmark_X,axis=1)

        else:
            benchmark_weights = weights[:, start_index:end_index].reshape(1, benchmark_size)

        benchmark_Y = Y[:, start_index:end_index].reshape(1, benchmark_size)

        benchmarks_X.append(benchmark_X.T.flatten())
        benchmarks_Y.append(benchmark_Y)
        benchmarks_weights.append(benchmark_weights)

    # if generate_weight and not unit_weight and add_weights:
    #     X = np.vstack((X, benchmark_noisy_weights.flatten()))
    X = np.delete(X, 0, 0)

    dataset = {'X': X,
               'Y': Y,
               'weights': weights,
               'benchmarks_X': benchmarks_X,
               'benchmarks_Y': benchmarks_Y,
               'benchmarks_weights': benchmarks_weights}
    return dataset
