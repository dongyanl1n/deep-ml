import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	batches = []
	num_batches = len(X) // batch_size
	if len(X) % batch_size != 0:
		num_batches += 1
	for i_batch in range(num_batches):
		start = i_batch * batch_size
		end = min((i_batch+1) * batch_size, len(X))
		batch = [X[start:end]]
		if y is not None:
			batch.append(y[start:end])
		batches.append(batch)
	return batches
