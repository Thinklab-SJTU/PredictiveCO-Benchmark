import csv
import os

import numpy as np


################################# File utils ################################################
def read_file(filename, folder_path="data", delimiter=" "):
    """
    read the dataset with filename and return the feature and labels in list.
    :param filename (str): filename of the dataset
    :return: data(list) : all features and labels in the data. We transform the data into features and labels with a different method, unique for each dataset.
    """
    file_path = get_file_path(
        filename, folder_path=os.path.join(os.getcwd(), folder_path)
    )

    with open(file_path, "r") as f:
        data = list(csv.reader(f, delimiter=delimiter))
    return data


def get_file_path(filename, folder_path="data"):
    """
    Constructs filepath. dataset is expected to be in the "data" folder
    :param filename:
    :return:
    """
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(dir_path, folder_path, filename)
    return file_path


################################# Problem utils ################################################
def generate_uniform_weights_from_seed(benchmark_size, weight_seed):
    number_of_each_weight = int(benchmark_size / len(weight_seed))
    uniform_weights_from_seed = np.array(
        [np.ones((number_of_each_weight)) * weight for weight in weight_seed]
    ).flatten()
    np.random.shuffle(uniform_weights_from_seed)
    uniform_weights_from_seed = uniform_weights_from_seed.reshape(
        (1, uniform_weights_from_seed.size)
    )

    return uniform_weights_from_seed
