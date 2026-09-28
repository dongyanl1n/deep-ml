import numpy as np
def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	# Your code here
	bce_loss = []
	for y, y_p in zip(y_true, y_pred):
		l = y * np.log(y_p - epsilon) + (1-y) * np.log(1-y_p - epsilon)
		bce_loss.append(-l)
	return np.mean(np.array(bce_loss))