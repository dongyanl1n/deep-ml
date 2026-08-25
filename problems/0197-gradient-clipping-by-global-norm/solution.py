import numpy as np
def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
	"""
	Clip gradients by global norm.
	
	Args:
		gradients: List of gradient arrays
		max_norm: Maximum allowed global norm
	
	Returns:
		List of clipped gradient arrays
	"""
	# Your code here
	# Need to consider the edge case where not all lists in gradients have the same lengths
	arrays = [np.array(grad, dtype=float) for grad in gradients]
    
    # sum of squares across all params, then sqrt, gives the global norm
    global_norm = np.sqrt(sum(np.sum(arr ** 2) for arr in arrays))
    
    if global_norm > max_norm:
        scaling_factor = max_norm / global_norm
        return [(arr * scaling_factor).tolist() for arr in arrays]
    else:
        return [arr.tolist() for arr in arrays]