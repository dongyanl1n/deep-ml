import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def swish(x):
        return np.multiply(x, sigmoid(x))
    d = int(x.shape[1] // 2)
    x1 = x[:, :d]
    x2 = x[:, d:]
    scores = np.multiply(x1, swish(x2))
    return scores