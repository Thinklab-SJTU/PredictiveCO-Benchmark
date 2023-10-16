import numpy as np
import torch


def move_to_array(Y):
    if torch.is_tensor(Y):
        Y_array = Y.detach().cpu().numpy()
    else:
        Y_array = np.array(Y)
    return Y_array


def move_to_tensor(Y):
    if torch.is_tensor(Y):
        Y_tensor = Y
    else:
        Y_tensor = torch.from_numpy(np.array(Y).astype(np.float32))
    return Y_tensor


def get_idxs(obj, idxs):
    if torch.is_tensor(obj):
        return obj[[idxs]]
    elif isinstance(obj, list):
        return obj[idxs].unsqueeze(0)


def rand_like(obj, device="cpu"):
    if torch.is_tensor(obj):
        return torch.rand_like(obj, device=device)
    elif isinstance(obj, list):
        rand_obj = list()
        for idx in range(len(obj)):
            rand_obj.append(torch.rand_like(obj[idx], device=device))
        return rand_obj


def minus(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a - b
    elif isinstance(a, list) and isinstance(b, list):
        a, b = torch.stack(a), torch.stack(b)
        return a - b
