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
        rand_seed=0,  # for reproducibility
        version="warcraft",
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(Shortestpath, self).__init__()
        self._set_seed(rand_seed)
        self.size = size
        if version == "warcraft":
            self.load_dataset(
                data_dir + f"/{size}x{size}/",
                use_test_set=True,
                evaluate_with_extra=False,
                normalize=True,
            )

    def load_dataset(self, data_dir, use_test_set, evaluate_with_extra, normalize):
        train_prefix = "train"
        data_suffix = "maps"
        val_prefix = "val"
        test_prefix = "test"
        # val_prefix = ("test" if use_test_set else "val") + (
        #     "_extra" if evaluate_with_extra else ""
        # )
        train_data_path = os.path.join(
            data_dir, train_prefix + "_" + data_suffix + ".npy"
        )

        if not os.path.exists(train_data_path):
            raise Exception(f"Cannot find {train_data_path}")

        def read_data(split_prefix):
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
                mean, std = (
                    np.mean(inputs, axis=(0, 2, 3), keepdims=True),
                    np.std(inputs, axis=(0, 2, 3), keepdims=True),
                )
            inputs -= mean
            inputs /= std
            return (
                torch.FloatTensor(inputs),
                torch.FloatTensor(labels).reshape(len(labels), -1),
                torch.FloatTensor(true_weights).reshape(len(labels), -1),
                full_images,
            )

        (self.train_inputs, self.train_labels, self.train_true_weights, _) = read_data(
            train_prefix
        )
        self.val_inputs, self.val_labels, self.val_true_weights, _ = read_data(val_prefix)
        self.test_inputs, self.test_labels, self.test_true_weights, _ = read_data(
            test_prefix
        )
        print(
            self.train_inputs.shape,
            self.train_labels.shape,
            self.train_true_weights.shape,
        )
        # (10000, 3, 96, 96) (10000, 12, 12) (10000, 12, 12)
        # print(self.val_inputs.shape, self.val_labels.shape)
        # (1000, 3, 96, 96) (1000, 12, 12) (1000, 12, 12)

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
        return self.train_inputs, self.train_true_weights, self.train_labels

    def get_val_data(self, **kwargs):
        return self.val_inputs, self.val_true_weights, self.val_labels

    def get_test_data(self, **kwargs):
        return self.test_inputs, self.test_true_weights, self.test_labels

    def get_model_shape(self):
        assert self.train_inputs.shape[-1] == 8 * self.size
        return self.train_inputs.shape[-1], self.size**2

    def get_output_activation(self):
        return "sigmoid"

    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
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
        Y = to_device(Y, "cpu")
        return (Y * Z).sum(-1)

    def get_twostageloss(self):
        return "bce"

    def init_API(self):
        return {"modelSense": GRB.MAXIMIZE, "n_vars": self.size**2, "size": self.size}
