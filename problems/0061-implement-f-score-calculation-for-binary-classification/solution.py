import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	pred_true = np.zeros((len(y_pred), 2))
	pred_true[:, 0] = y_pred
	pred_true[:, 1] = y_true

	true_positive = np.all(pred_true==np.array((1,1)), axis=1).sum()
	false_positive = np.all(pred_true==np.array((1,0)), axis=1).sum()
	true_negative = np.all(pred_true==np.array((0,0)), axis=1).sum()
	false_negative = np.all(pred_true==np.array((0,1)), axis=1).sum()
	# print(true_positive, true_negative)
	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / (true_positive + false_negative)

	f_score = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall)

	return np.round(f_score, 3)
