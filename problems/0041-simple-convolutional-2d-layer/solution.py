import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# initialize the shape of output
	output_height = (input_height + 2*padding - kernel_height) // stride + 1
	output_width = (input_width + 2*padding - kernel_width) // stride + 1
	output_matrix = np.zeros((output_height, output_width))
	padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)))
	padded_height, padded_width = padded_input.shape

	# for (x, i_h) in enumerate(range(0, padded_height, stride)):
	# 	for (y, i_w) in enumerate(range(0, padded_width, stride)):
	# 		output_matrix[x ,y] = np.sum(padded_input[i_h:i_h+kernel_height, i_w:i_w+kernel_width] * kernel)
	# The above won't work if the division doesn't work out evenly
	for x in range(output_height):
		for y in range(output_width):
			i_h = x * stride
			i_w = y * stride
			output_matrix[x, y] = np.sum(
				padded_input[i_h:i_h+kernel_height, i_w:i_w+kernel_width] * kernel
			)
	return output_matrix
