import numpy as np
import torch


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [ob.to(device) for ob in obj]
    else:
        return obj


def to_array(Y):
    if torch.is_tensor(Y):
        Y_array = Y.detach().cpu().numpy()
    elif isinstance(Y, list):
        Y_array = list()
        for item in Y:
            if torch.is_tensor(item):
                Y_array.append(item.detach().cpu().numpy())
            else:
                Y_array.append(item)
    else:
        Y_array = Y
    return Y_array


def to_tensor(Y):
    if torch.is_tensor(Y):
        Y_tensor = Y
    elif isinstance(Y, list):
        Y_tensor = list()
        for item in Y:
            if torch.is_tensor(item):
                Y_tensor.append(torch.from_numpy(np.array(item).astype(np.float32)))
            else:
                Y_tensor.append(item)
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
