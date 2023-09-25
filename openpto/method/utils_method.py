import numpy as np
import torch


def move_to_array(Y):
    if torch.is_tensor(Y):
        Y_array = Y.detach().cpu().numpy()
    else:
        Y_array = Y
    return Y_array


def move_to_tensor(Y):
    if torch.is_tensor(Y):
        Y_tensor = Y
    else:
        Y_tensor = torch.from_numpy(Y.astype(np.float32))
    return Y_tensor
