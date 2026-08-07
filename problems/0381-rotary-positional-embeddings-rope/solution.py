import numpy as np


def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape
    half_d = d // 2

    # Frequencies: theta_i = 1 / base^(2i/d), for i = 0, ..., d/2 - 1
    i = np.arange(half_d)
    theta = 1.0 / (base ** (2 * i / d))          # shape (d/2,)

    # Angles: position * frequency, for every (position, pair) combo
    angles = positions[:, None].astype(np.float64) * theta[None, :]   # shape (seq_len, d/2)

    cos = np.cos(angles)   # (seq_len, d/2)
    sin = np.sin(angles)   # (seq_len, d/2)

    # Split embedding into even/odd interleaved dimensions
    x_even = x[:, 0::2]    # (seq_len, d/2)
    x_odd  = x[:, 1::2]    # (seq_len, d/2)

    # Apply 2D rotation to each pair
    new_even = x_even * cos - x_odd * sin
    new_odd  = x_even * sin + x_odd * cos

    # Interleave back into original shape
    out = np.empty_like(x, dtype=np.float64)
    out[:, 0::2] = new_even
    out[:, 1::2] = new_odd

    return out
    
