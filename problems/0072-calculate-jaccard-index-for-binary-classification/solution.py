
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
	if np.all(y_true==0) and np.all(y_pred)==0:
		raise ValueError("both arrays contain only zeros")

	
	tp = np.sum((y_true==1) & (y_pred==1))
	fp = np.sum((y_true==0) & (y_pred==1))
	fn = np.sum((y_true==1) & (y_pred==0))
	result = tp / (tp+fn+fp)
	return round(result, 3)
