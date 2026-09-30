import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	np.random.seed(seed)
	ordering = np.random.permutation(len(X))
	return X[ordering], y[ordering]