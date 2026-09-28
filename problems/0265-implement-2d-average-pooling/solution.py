import numpy as np
def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	# Your code here
	# pass
	input_matrix = np.array(input_matrix)
	H, W = input_matrix.shape
	output_matrix = np.zeros((H//pool_size, W//pool_size))
	for h_i in range(H//pool_size):
		for w_i in range(W//pool_size):
			h_start, h_end = h_i * pool_size, h_i * pool_size + pool_size
			w_start, w_end = w_i * pool_size, w_i * pool_size + pool_size
			output_matrix[h_i, w_i] = np.mean(input_matrix[h_start:h_end, w_start:w_end])
	return output_matrix.tolist()