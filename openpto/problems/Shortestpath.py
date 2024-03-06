import os

import numpy as np
import torch


# from decorators import input_to_numpy
# from utils import TrainingIterator
from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.utils_method import to_array, to_device, to_tensor
from openpto.problems.PTOProblem import PTOProblem


class Shortestpath(PTOProblem):
    """ """

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for validation
        size=12,
        normalize=True,
        rand_seed=0,  # for reproducibility
        prob_version="warcraft",
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(Shortestpath, self).__init__()
        self._set_seed(rand_seed)
        self.size = size
        self.prob_version = prob_version
        if prob_version == "warcraft":
            self.load_dataset(
                data_dir + f"/{size}x{size}/",
                use_test_set=True,
                evaluate_with_extra=False,
                normalize=normalize,
            )
        elif prob_version == "direct":
            self.load_dataset(
                data_dir + f"/{size}x{size}/",
                use_test_set=True,
                evaluate_with_extra=False,
                normalize=normalize,
            )
        else:
            raise NotImplementedError

    def load_dataset(self, data_dir, use_test_set, evaluate_with_extra, normalize):
        data_suffix = "maps"
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        def read_data(split_prefix, normalize):
            inputs = np.load(
                os.path.join(data_dir, split_prefix + "_" + data_suffix + ".npy")
            ).astype(np.float32)
            # noinspection PyTypeChecker
            inputs = inputs.transpose(0, 3, 1, 2)  # channel first

            labels = np.load(os.path.join(data_dir, split_prefix + "_shortest_paths.npy"))
            true_weights = np.load(
                os.path.join(data_dir, split_prefix + "_vertex_weights.npy")
            )
            full_images = np.load(os.path.join(data_dir, split_prefix + "_maps.npy"))
            if normalize:
                in_mean, in_std = (
                    np.mean(inputs, axis=(0, 2, 3), keepdims=True),
                    np.std(inputs, axis=(0, 2, 3), keepdims=True),
                )
                inputs = (inputs - in_mean) / in_std
            # normalize the weights
            if self.prob_version != "direct" and normalize:
                wei_mean, wei_std = (
                    np.mean(true_weights, axis=(0, 1, 2), keepdims=True),
                    np.std(true_weights, axis=(0, 1, 2), keepdims=True),
                )
                true_weights = (true_weights - wei_mean) / wei_std
            return (
                torch.FloatTensor(inputs),
                torch.FloatTensor(labels).reshape(len(labels), -1),
                torch.FloatTensor(true_weights).reshape(len(labels), -1),
                full_images,
            )

        (self.train_inputs, self.train_labels, self.train_true_weights, _) = read_data(
            train_prefix, normalize
        )  # (10000, 3, 96, 96) (10000, 12 * 12) (10000, 12 * 12)
        self.val_inputs, self.val_labels, self.val_true_weights, _ = read_data(
            val_prefix, normalize
        )  # (1000, 3, 96, 96) (1000, 12 * 12) (1000, 12 * 12)
        self.test_inputs, self.test_labels, self.test_true_weights, _ = read_data(
            test_prefix, normalize
        )
        print(
            self.train_inputs.shape,
            self.train_labels.shape,
            self.train_true_weights.shape,
        )

        # # @input_to_numpy
        # def denormalize(x):
        #     return (x * std) + mean
        # metadata = {
        #     "input_image_size": val_full_images[0].shape[1],
        #     "output_features": val_true_weights[0].shape[0]
        #                     * val_true_weights[0].shape[1],
        #     "num_channels": val_full_images[0].shape[-1],
        #     "denormalize": denormalize,
        # }
        return

    # def get_train_sol(self):
    #     return self.train_labels

    # def get_val_sol(self):
    #     return self.val_labels

    # def get_test_sol(self):
    #     return self.test_labels

    def get_train_data(self, **kwargs):
        if self.prob_version == "direct":
            return self.train_inputs, self.train_labels, self.train_true_weights
        else:
            return self.train_inputs, self.train_true_weights, self.train_labels

    def get_val_data(self, **kwargs):
        if self.prob_version == "direct":
            return self.val_inputs, self.val_labels, self.val_true_weights
        else:
            return self.val_inputs, self.val_true_weights, self.val_labels

    def get_test_data(self, **kwargs):
        if self.prob_version == "direct":
            return self.test_inputs, self.test_labels, self.test_true_weights
        else:
            return self.test_inputs, self.test_true_weights, self.test_labels

    def get_model_shape(self):
        assert self.train_inputs.shape[-1] == 8 * self.size
        return self.train_inputs.shape[-1], self.size**2

    def get_eval_metric(self):
        return "match"

    def get_output_activation(self):
        if self.prob_version == "direct":
            return "sigmoid"
        else:
            return "identity"

    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
        if self.prob_version == "direct":
            sol = Y.round()
            obj = sol
            return sol, obj
        else:
            Y = to_device(Y, "cpu")
            sol = []
            for i in range(len(Y)):
                # solve
                solp, other = optSolver.solve(to_array(Y[i]))
                sol.append(solp)
            sol = to_tensor(np.array(sol))
            obj = self.get_objective(Y, sol, kwargs)
        return sol, obj

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        if self.prob_version == "direct":
            return Z
        else:
            Z = to_device(Z, Y.device)
            return (Y * Z).sum(-1)

    def get_twostageloss(self):
        if self.prob_version == "direct":
            return "bce"
        else:
            return "mse"

    def init_API(self):
        if self.prob_version == "direct":
            return {
                "modelSense": GRB.MINIMIZE,
                "n_vars": self.size**2,
                "size": self.size,
            }
        else:
            return {
                "modelSense": GRB.MAXIMIZE,
                "n_vars": self.size**2,
                "size": self.size,
            }
