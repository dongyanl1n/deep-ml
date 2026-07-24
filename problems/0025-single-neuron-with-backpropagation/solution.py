import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	n = len(labels) # n_examples
	mse_values = []

	for i_epoch in range(epochs):
		activation = features @ initial_weights + initial_bias
		output = sigmoid(activation)
		# mse_values = (output - labels)**2
		mse = np.mean((output - labels) ** 2)
		mse_values.append(round(mse, 4))
		# dloss_dw = 2*(output-labels) * (output*(1-output)) @ features.T
		# dloss_db = 2*(output-labels) * (output*(1-output))
		delta = 2*(output-labels) * (output*(1-output)) / n  # (n_examples,)
		dloss_db = np.sum(delta)   # scalar
		dloss_dw = features.T @ delta # (n_features,)
		# features: (n_examples, n_features)
		updated_weights = initial_weights  - learning_rate * dloss_dw
		updated_bias = initial_bias - learning_rate * dloss_db
		initial_weights = updated_weights
		initial_bias = updated_bias

	return updated_weights, updated_bias, mse_values