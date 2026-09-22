import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                        num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    B, C, H, W = X.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"

    # (B, G, C/G, H, W): each group becomes its own axis
    Xg = X.reshape(B, num_groups, C // num_groups, H, W)

    # one mean/var per (sample, group), pooled over channels-in-group and spatial dims
    mean = Xg.mean(axis=(2, 3, 4), keepdims=True)
    var = Xg.var(axis=(2, 3, 4), keepdims=True)
    Xg_norm = (Xg - mean) / np.sqrt(var + epsilon)

    X_norm = Xg_norm.reshape(B, C, H, W)

    # accept gamma/beta as either shape (C,) or (1, C, 1, 1)
    gamma = np.asarray(gamma).reshape(1, C, 1, 1)
    beta = np.asarray(beta).reshape(1, C, 1, 1)
    return gamma * X_norm + beta