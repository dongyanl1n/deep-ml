import numpy as np
import math

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
    	Predicted class labels for X_test (shape: N_test)
	"""
    N_train, D = X_train.shape
    classes, counts = np.unique(y_train, return_counts=True)
    n_classes = len(classes)
    prior_prob = counts / np.sum(counts)

    X_train_mean = np.zeros((n_classes, D))
    X_train_var = np.zeros((n_classes, D))
    for i_class, classs in enumerate(classes):
        X_train_mean[i_class] = np.mean(X_train[y_train == classs], axis=0)
        X_train_var[i_class] = np.var(X_train[y_train == classs], axis=0) + 1e-9

    y_test = np.zeros(X_test.shape[0], dtype=classes.dtype)
    for i_test, x_test in enumerate(X_test):
        log_posteriors = np.zeros(n_classes)
        for i_class in range(n_classes):
            log_likelihood_per_feature = (
                -0.5 * np.log(2 * math.pi * X_train_var[i_class])
                - (x_test - X_train_mean[i_class])**2 / (2 * X_train_var[i_class])
            )
            log_posteriors[i_class] = np.log(prior_prob[i_class]) + np.sum(log_likelihood_per_feature)
        y_test[i_test] = classes[np.argmax(log_posteriors)]

    return y_test
