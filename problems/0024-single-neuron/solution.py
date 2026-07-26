import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))
	features, labels, weights = np.array(features), np.array(labels), np.array(weights)
	probabilities = [sigmoid(prob+bias) for prob in (features @ weights)]
	probabilities = [round(n, 4) for n in probabilities]
	mse = np.mean((probabilities-labels)**2)
	mse = round(mse, 4)
	return probabilities, mse