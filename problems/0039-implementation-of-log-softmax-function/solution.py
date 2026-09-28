import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	score = np.array(scores)
	scores_max = np.max(scores)
	expo = scores - scores_max
	numerator = np.exp(expo)
	denominator = sum(numerator)
	softmax_scores = numerator / denominator
	log_softmax_scores = np.log(softmax_scores)
	return log_softmax_scores