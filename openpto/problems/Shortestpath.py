import os

import numpy as np
import torch


# from decorators import input_to_numpy
# from utils import TrainingIterator
from gurobipy import GRB  # pylint: disable=no-name-in-module
from torchvision import transforms as transforms

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
                normalize=normalize,
            )
        elif prob_version == "warcraft-ood":
            self.load_ood_dataset(
                data_dir + f"/{size}x{size}/",
                normalize=normalize,
            )
        elif prob_version == "direct":
            self.load_dataset(
                data_dir + f"/{size}x{size}/",
                normalize=normalize,
            )
        else:
            raise NotImplementedError

    def read_data(self, data_dir, split_prefix, normalize):
        data_suffix = "maps"
        inputs = np.load(
            os.path.join(data_dir, split_prefix + "_" + data_suffix + ".npy")
        ).astype(np.float32)
        # channel last
        # inputs = inputs.transpose(0, 3, 1, 2)  # channel first

        labels = np.load(os.path.join(data_dir, split_prefix + "_shortest_paths.npy"))
        true_weights = np.load(
            os.path.join(data_dir, split_prefix + "_vertex_weights.npy")
        )
        full_images = np.load(os.path.join(data_dir, split_prefix + "_maps.npy"))
        print("inputs: ", inputs.shape, "true_weights: ", true_weights.shape)
        if normalize:
            in_mean, in_std = (
                np.mean(inputs, axis=(0, 1, 2), keepdims=True),
                np.std(inputs, axis=(0, 1, 2), keepdims=True),
            )
            inputs = (inputs - in_mean) / in_std
        # normalize the weights
        # if normalize:  # true_weights:  (10000, 12, 12)
        #     wei_mean, wei_std = (
        #         np.mean(true_weights, axis=(0, 1, 2), keepdims=True),
        #         np.std(true_weights, axis=(0, 1, 2), keepdims=True),
        #     )
        #     true_weights = (true_weights - wei_mean) / wei_std
        return (
            torch.FloatTensor(inputs),
            torch.FloatTensor(labels).reshape(len(labels), -1),
            torch.FloatTensor(true_weights).reshape(len(labels), -1),
            full_images,
        )

    def load_dataset(self, data_dir, normalize):
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        (self.train_X, self.train_Y, self.train_weights, _) = self.read_data(
            data_dir, train_prefix, normalize
        )  # (10000, 3, 96, 96) (10000, 12 * 12) (10000, 12 * 12)
        self.val_X, self.val_Y, self.val_weights, _ = self.read_data(
            data_dir, val_prefix, normalize
        )  # (1000, 3, 96, 96) (1000, 12 * 12) (1000, 12 * 12)
        self.test_X, self.test_Y, self.test_weights, _ = self.read_data(
            data_dir, test_prefix, normalize
        )
        print(
            "inputs, labels, weights: ",
            self.train_X.shape,
            self.train_Y.shape,
            self.train_weights.shape,
        )
        return

    def load_ood_dataset(self, data_dir, normalize):
        train_prefix, val_prefix, test_prefix = "train", "val", "test"

        ########## change distribution of data sub-function ################
        def transform_contrast(images, normalize):
            transform = transforms.Compose(
                [
                    transforms.ColorJitter(contrast=10),
                ]
            )
            transformed_images = transform(images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            if normalize:
                transformed_images = do_norm(transformed_images)
            return transformed_images

        ###################### read data sub-function ######################
        def do_norm(inputs):
            in_mean, in_std = (
                torch.mean(inputs, axis=(0, 1, 2), keepdims=True),
                torch.std(inputs, axis=(0, 1, 2), keepdims=True),
            )
            return (inputs - in_mean) / in_std

        ##### Read Data for ver0; as train distribution
        ver0_train_X, self.train_Y, self.train_weights, _ = self.read_data(
            data_dir, train_prefix, False
        )  # (10000, 3, 96, 96) (10000, 12 * 12) (10000, 12 * 12)
        ver0_val_X, self.val_Y, self.val_weights, _ = self.read_data(
            data_dir, val_prefix, False
        )
        ver0_test_X, self.test_Y, self.test_weights, _ = self.read_data(
            data_dir, test_prefix, False
        )
        ##### Out the original data
        self.ver0_train_X, self.ver0_val_X = (
            do_norm(ver0_train_X),
            do_norm(ver0_val_X),
        )
        ##### Pertrub the data, get ver1
        self.train_X, self.val_X, self.test_X = (
            transform_contrast(ver0_train_X, normalize),
            transform_contrast(ver0_val_X, normalize),
            transform_contrast(ver0_test_X, normalize),
        )
        return

    # def get_train_sol(self):
    #     return self.train_Y

    # def get_val_sol(self):
    #     return self.val_Y

    # def get_test_sol(self):
    #     return self.test_Y

    def get_train_data(self, train_mode="iid", **kwargs):
        if self.prob_version == "direct":
            return self.train_X, self.train_Y, self.train_weights
        else:
            if train_mode == "iid":
                return self.train_X, self.train_weights, self.train_Y
            elif train_mode == "ood":
                return self.ver0_train_X, self.train_weights, self.train_Y

    def get_val_data(self, train_mode="iid", **kwargs):
        if self.prob_version == "direct":
            return self.val_X, self.val_Y, self.val_weights
        else:
            if train_mode == "iid":
                return self.val_X, self.val_weights, self.val_Y
            elif train_mode == "ood":
                return self.ver0_val_X, self.val_weights, self.val_Y

    def get_test_data(self, train_mode="iid", **kwargs):
        if self.prob_version == "direct":
            return self.test_X, self.test_Y, self.test_weights
        else:
            return self.test_X, self.test_weights, self.test_Y

    def get_model_shape(self):
        assert self.train_X.shape[2] == 8 * self.size
        return self.train_X.shape[2], self.size**2

    def get_eval_metric(self):
        # return "match"
        return "regret"

    def get_output_activation(self):
        if self.prob_version == "direct":
            return "sigmoid"
        else:
            return "identity"

    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
        if self.prob_version == "direct":
            sol = Y.round()
            obj = self.get_objective(params, sol, kwargs)
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
            Z = to_device(Z, Y.device)  # 10000,144
            return torch.sum(Y * Z, -1, keepdim=True)

    def get_twostageloss(self):
        if self.prob_version == "direct":
            return "bce"
        else:
            return "mse"

    def init_API(self):
        return {
            "modelSense": GRB.MINIMIZE,
            "n_vars": self.size**2,
            "size": self.size,
        }

    def genEnv(
        self,
        env_id,
        num_train_instances,
    ):
        self.env_config[f"env{env_id}"]
        # print("config: ", config)
        Xs_train, Ys_train = None, None
        return Xs_train, Ys_train
