import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if n_col is None:
		n_col = np.max(x) + 1
	one_hot = np.zeros((len(x), n_col))
	indices = x
	for i in range(len(x)):
		one_hot[i, indices[i]] = 1
	return one_hot
