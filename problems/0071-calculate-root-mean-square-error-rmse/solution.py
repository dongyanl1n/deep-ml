
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	assert len(y_pred) == len(y_true), "mismatched array shape!"
	assert type(y_pred) == type(y_true) == np.ndarray, "invalid input types"
	assert len(y_pred) != 0  and len(y_true) != 0
	rmse_res = np.sqrt(np.mean((y_true - y_pred)**2))
	return round(rmse_res,3)
