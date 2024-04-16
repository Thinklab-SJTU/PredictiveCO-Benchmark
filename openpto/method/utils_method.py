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
        return Y.detach().cpu().numpy()
    elif isinstance(Y, list) and torch.is_tensor(Y[0]):
        return [item.detach().cpu().numpy() for item in Y]
    elif isinstance(Y, list) and isinstance(Y[0], np.ndarray):
        return [item.detach().cpu().numpy() for item in Y]
    else:
        return Y


def to_tensor(Y):
    if torch.is_tensor(Y):
        return Y
    elif isinstance(Y, list) and torch.is_tensor(Y[0]):
        return Y
    elif isinstance(Y, list) and isinstance(Y[0], np.ndarray):
        return [torch.from_numpy(np.array(item).astype(np.float32)) for item in Y]
    else:
        return torch.from_numpy(np.array(Y).astype(np.float32))


def get_idxs(obj, idxs):
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj[[idxs]]
    elif isinstance(obj, list):
        return obj[idxs].unsqueeze(0)


def get_batch(obj, idx, batch_size):
    begin = idx * batch_size
    end = (idx + 1) * batch_size
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj[begin:end]
    elif isinstance(obj, list):
        return obj[begin:end].unsqueeze(0)


def rand_like(obj, device="cpu"):
    if torch.is_tensor(obj):
        return torch.rand_like(obj, device=device)
    elif isinstance(obj, list):
        return [torch.rand_like(ob, device=device) for ob in obj]
    else:
        raise NotImplementedError()


def do_reduction(obj, reduction, dim=None):
    if reduction == "mean":
        if dim is None:
            obj = torch.mean(obj)
        else:
            obj = torch.mean(obj, dim=dim)
    elif reduction == "sum":
        if dim is None:
            obj = torch.sum(obj)
        else:
            obj = torch.sum(obj, dim=dim)
    elif reduction == "none":
        pass
    else:
        raise ValueError("No reduction '{}'.".format(reduction))
    return obj


def ndiv(a, b):
    if b == 0:
        return 0
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def minus(a, b):
    if (torch.is_tensor(a) and torch.is_tensor(b)) or (
        isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    ):
        return a - b
    elif isinstance(a, list) and isinstance(b, list):
        a, b = torch.stack(a), torch.stack(b)
        return a - b
    else:
        raise NotImplementedError
