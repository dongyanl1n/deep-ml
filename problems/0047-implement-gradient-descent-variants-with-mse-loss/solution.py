import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    # pass
    for i_epoch in range(n_epochs):
        # loss = (y - np.matmul(X, weights))**2

        if method == 'batch': # use all samples
            dloss_dw = X.T @ (X @ weights - y)
            weights -= 2/len(X) * dloss_dw * learning_rate
        elif method == 'stochastic':
            for i_sample in range(len(X)):
                dloss_dw = 2 * X[i_sample] * (X[i_sample] @ weights - y[i_sample])
                weights -= dloss_dw.T * learning_rate
        elif method == 'mini_batch':
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                dloss_dw = 2/batch_size * (X_batch.T @ (X_batch @ weights - y_batch))
                weights -= dloss_dw * learning_rate
    
    return weights




            
        

