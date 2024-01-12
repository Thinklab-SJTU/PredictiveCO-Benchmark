import os

import numpy as np

from decorators import input_to_numpy
from utils import TrainingIterator

from openpto.problems.PTOProblem import PTOProblem


class Shortestpath(PTOProblem):
    """ """

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_items=100,  # number of targets to consider
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        **kwargs,
    ):
        super(Shortestpath, self).__init__()
        self._set_seed(rand_seed)

    @staticmethod
    def gendata():
        return

    @staticmethod
    def load_dataset(
        data_dir, use_test_set, evaluate_with_extra, normalize, use_local_path
    ):
        train_prefix = "train"
        data_suffix = "maps"

        val_prefix = ("test" if use_test_set else "val") + (
            "_extra" if evaluate_with_extra else ""
        )
        train_data_path = os.path.join(
            data_dir, train_prefix + "_" + data_suffix + ".npy"
        )

        if os.path.exists(train_data_path):
            train_inputs = np.load(
                os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy")
            ).astype(np.float32)
            # noinspection PyTypeChecker
            train_inputs = train_inputs.transpose(0, 3, 1, 2)  # channel first

            train_labels = np.load(
                os.path.join(data_dir, train_prefix + "_shortest_paths.npy")
            )
            train_true_weights = np.load(
                os.path.join(data_dir, train_prefix + "_vertex_weights.npy")
            )
            if normalize:
                mean, std = (
                    np.mean(train_inputs, axis=(0, 2, 3), keepdims=True),
                    np.std(train_inputs, axis=(0, 2, 3), keepdims=True),
                )
            train_inputs -= mean
            train_inputs /= std
            train_iterator = TrainingIterator(
                dict(
                    images=train_inputs,
                    labels=train_labels,
                    true_weights=train_true_weights,
                )
            )
        else:
            raise Exception(f"Cannot find {train_data_path}")

        val_inputs = np.load(
            os.path.join(data_dir, val_prefix + "_" + data_suffix + ".npy")
        ).astype(np.float32)
        # noinspection PyTypeChecker
        val_inputs = val_inputs.transpose(0, 3, 1, 2)  # channel first

        if normalize:
            # noinspection PyUnboundLocalVariable
            val_inputs -= mean
            # noinspection PyUnboundLocalVariable
            val_inputs /= std

        val_labels = np.load(os.path.join(data_dir, val_prefix + "_shortest_paths.npy"))
        val_true_weights = np.load(
            os.path.join(data_dir, val_prefix + "_vertex_weights.npy")
        )
        val_full_images = np.load(os.path.join(data_dir, val_prefix + "_maps.npy"))
        eval_iterator = TrainingIterator(
            dict(images=val_inputs, labels=val_labels, true_weights=val_true_weights)
        )

        @input_to_numpy
        def denormalize(x):
            return (x * std) + mean

        metadata = {
            "input_image_size": val_full_images[0].shape[1],
            "output_features": val_true_weights[0].shape[0]
            * val_true_weights[0].shape[1],
            "num_channels": val_full_images[0].shape[-1],
            "denormalize": denormalize,
        }

        return train_iterator, eval_iterator, metadata

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
