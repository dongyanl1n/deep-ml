import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
	return np.array([x_i if x_i > 0 else 0 for x_i in x])
	# return np.max(x, 0)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	a1 = relu(w1 @ x)
	a2 = w2 @ a1
	return relu(x + a2)
