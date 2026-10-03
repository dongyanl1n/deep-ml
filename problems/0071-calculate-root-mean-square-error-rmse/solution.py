import numpy as np

def rmse(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("inputs must be array-like and numeric")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("inputs must be non-empty")

    return round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3)
