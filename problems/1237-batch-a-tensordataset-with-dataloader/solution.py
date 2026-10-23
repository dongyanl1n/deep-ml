import torch
from torch.utils.data import TensorDataset, DataLoader
import math

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    data = TensorDataset(X, y)
    dataloader = DataLoader(data, batch_size=4, shuffle=False)
    data = next(iter(dataloader))
    num_batches = math.ceil(len(data) / 4)
    first_batch_X_shape_tuple = (4, X.shape[-1])
    return (num_batches, first_batch_X_shape_tuple)

