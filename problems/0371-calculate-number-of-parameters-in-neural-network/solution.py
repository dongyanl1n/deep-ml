def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	# Your code here
	total_param = 0
	for layer in layers:
		if layer['type'] == 'dense':
			param = layer['input_size'] * layer['output_size']
			if 'bias' not in layer or layer['bias']:
				param += layer['output_size']
		elif layer['type'] == 'conv2d':
			param = layer['in_channels'] * layer['out_channels'] * layer['kernel_size']**2
			if 'bias' not in layer or layer['bias']:
				param += layer['out_channels']
		total_param += param
	return int(total_param)