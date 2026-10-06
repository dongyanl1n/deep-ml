import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    # pass
    a = np.min(values)
    b = np.max(values)
    new_values = c + (d-c)/(b-a) * (values - a)
    return new_values