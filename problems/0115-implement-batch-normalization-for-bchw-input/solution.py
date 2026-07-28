import numpy as np

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.
    
    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics (following PyTorch convention)
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics
    
    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    # Your code here
    # pass
    if training:
        batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # one mean per channel
        batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)
        X = (X - batch_mean) / np.sqrt(batch_var + epsilon)
        X = gamma * X + beta
        if running_mean is None:
            running_mean = batch_mean
        else:
            running_mean = (1-momentum)*running_mean + momentum * batch_mean
        
        if running_var is None:
            running_var = batch_var
        else:
            running_var = (1-momentum)*running_var + momentum * batch_var
    else:
        X = (X - running_mean) / np.sqrt(running_var + epsilon)
        X = gamma * X + beta
    return (X, running_mean, running_var)

