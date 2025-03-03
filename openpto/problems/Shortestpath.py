import glob
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
        num_train_instances=400,  # number of instances to use from the dataset to train
        num_test_instances=200,  # number of instances to use from the dataset to test
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
        self.normalize = normalize
        self.prob_version = prob_version
        # split
        self.n_vals = int(num_train_instances * val_frac)
        self.n_trains = num_train_instances - self.n_vals
        self.n_tests = num_test_instances
        ###
        if prob_version == "warcraft":
            self.load_dataset(
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

    @staticmethod
    def do_norm(inputs):
        in_mean, in_std = (
            torch.mean(inputs, axis=(0, 1, 2), keepdims=True),
            torch.std(inputs, axis=(0, 1, 2), keepdims=True),
        )
        # epsilon = 1e-6
        # in_std[in_std == 0] = epsilon
        return (inputs - in_mean) / in_std

    @staticmethod
    def read_npy_files(directory, prefix):
        """
        读取以指定前缀开头、以 partx 为后缀的 .npy 文件。
        """
        data_path = os.path.join(directory, prefix + ".npy")
        if os.path.exists(data_path):
            outputs = np.load(data_path).astype(np.float32)
        else:
            # 构造文件查找模式
            file_pattern = os.path.join(directory, f"{prefix}_part*.npy")
            file_list = glob.glob(file_pattern)
            print("file_pattern: ", file_pattern, "file_list: ", file_list)
            outputs_list = []
            for file_path in file_list:
                if os.path.exists(file_path):
                    outputs_list.append(np.load(file_path).astype(np.float32))
            outputs = np.vstack(outputs_list)
        return outputs

    def read_data(self, data_dir, split_prefix, normalize):
        data_suffix = "maps"
        inputs = self.read_npy_files(data_dir, split_prefix + "_" + data_suffix)
        # channel last
        # inputs = inputs.transpose(0, 3, 1, 2)  # channel first

        labels = self.read_npy_files(data_dir, split_prefix + "_shortest_paths")
        Y = self.read_npy_files(data_dir, split_prefix + "_vertex_weights")
        full_images = self.read_npy_files(data_dir, split_prefix + "_maps")
        # print("inputs: ", inputs.shape, "Y: ", Y.shape) #inputs:  (10000, 96, 96, 3) Y:  (10000, 12, 12)
        if normalize:
            inputs = self.do_norm(inputs)

        return (
            torch.FloatTensor(inputs),
            torch.FloatTensor(labels).reshape(len(labels), -1),
            torch.FloatTensor(Y).reshape(len(labels), -1),
            full_images,
        )

    def load_dataset(self, data_dir, normalize):
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"
        #

        train_X, train_Z, train_Y, _ = self.read_data(
            data_dir, train_prefix, normalize
        )  # (10000, 3, 96, 96) (10000, 12 * 12) (10000, 12 * 12)
        self.train_X, self.train_Z, self.train_Y = (
            train_X[: self.n_trains],
            train_Z[: self.n_trains],
            train_Y[: self.n_trains],
        )
        val_X, val_Z, val_Y, _ = self.read_data(
            data_dir, val_prefix, normalize
        )  # (1000, 3, 96, 96) (1000, 12 * 12) (1000, 12 * 12)
        self.val_X, self.val_Z, self.val_Y = (
            val_X[: self.n_vals],
            val_Z[: self.n_vals],
            val_Y[: self.n_vals],
        )
        test_X, test_Z, test_Y, _ = self.read_data(data_dir, test_prefix, normalize)
        self.test_X, self.test_Z, self.test_Y = (
            test_X[: self.n_tests],
            test_Z[: self.n_tests],
            test_Y[: self.n_tests],
        )
        print(
            "inputs, labels, weights: ",
            self.train_X.shape,
            self.train_Z.shape,
            self.train_Y.shape,
        )
        return

    def get_perturbed_data(self, transform, normalize):
        new_train_X, new_val_X, new_test_X = (
            self.augment_transform(self.ver0_train_X, transform, normalize),
            self.augment_transform(self.ver0_val_X, transform, normalize),
            self.augment_transform(self.ver0_test_X, transform, normalize),
        )
        return new_train_X, new_val_X, new_test_X

    @staticmethod
    def get_augmentation(aug_name, value):
        if aug_name == "contrast":
            aug_operator = transforms.ColorJitter(contrast=value)
        elif aug_name == "brightness":
            aug_operator = transforms.ColorJitter(brightness=value)
        elif aug_name == "saturation":
            aug_operator = transforms.ColorJitter(saturation=value)
        elif aug_name == "hue":
            aug_operator = transforms.ColorJitter(hue=value)
        else:
            raise NotImplementedError(aug_name)
        transform = transforms.Compose(
            [transforms.ToPILImage(), aug_operator, transforms.ToTensor()]
        )
        return transform

    ########## sub-function: change distribution of data ################
    def augment_transform(self, images, transform, normalize):
        transformed_images = torch.stack(
            [transform(im) for im in images.permute(0, 3, 1, 2)]
        )
        transformed_images = transformed_images.permute(0, 2, 3, 1)
        if normalize:
            transformed_images = self.do_norm(transformed_images)
        return transformed_images

    # def get_train_sol(self):
    #     return self.train_Z

    # def get_val_sol(self):
    #     return self.val_Z

    # def get_test_sol(self):
    #     return self.test_Z

    def get_train_data(self, train_mode="iid", **kwargs):
        if self.prob_version == "direct":
            return self.train_X, self.train_Z, self.train_Y
        else:
            if train_mode == "iid":
                return self.train_X, self.train_Y, self.train_Z

    def get_val_data(self, train_mode="iid", **kwargs):
        if self.prob_version == "direct":
            return self.val_X, self.val_Z, self.val_Y
        else:
            if train_mode == "iid":
                return self.val_X, self.val_Y, self.val_Z

    def get_test_data(self, train_mode="iid", **kwargs):
        if self.prob_version == "direct":
            return self.test_X, self.test_Z, self.test_Y
        else:
            return self.test_X, self.test_Y, self.test_Z

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

    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        if self.prob_version == "direct":
            sol = Y.round()
            obj = self.get_objective(params, sol, kwargs)
            return sol, obj
        else:
            Y = to_device(Y, "cpu")
            sol = []
            for i in range(len(Y)):
                # solve
                solp, other = ptoSolver.solve(to_array(Y[i]), **kwargs)
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
        do_debug=False,
        **kwargs,
    ):
        config = self.env_config[f"env{env_id}"]
        # print("config: ", config)
        ver1_transform = self.get_augmentation(config["type"], config["value"])
        new_train_X = self.augment_transform(
            to_device(self.ver0_train_X, "cpu"), ver1_transform, self.normalize
        )
        if do_debug:
            print("new_train_X: ", new_train_X)
            if torch.isnan(new_train_X).any():
                print("envs: ", env_id)
                print("input", new_train_X[0])
                assert 0
        # print("input before norm: ", new_train_X[0])
        # new_train_X = self.do_norm(new_train_X)
        # print("input after norm: ", new_train_X[0])
        return new_train_X, self.train_Y
