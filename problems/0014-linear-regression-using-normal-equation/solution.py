import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X)
	y = np.array(y)
	# Your code here, make sure to round
	theta = np.linalg.inv(X.T @ X) @ X.T @ y
	# theta = list(theta)
	theta = [round(t, 4) for t in theta]
	return theta