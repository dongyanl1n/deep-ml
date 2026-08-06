import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    # pass
    # n_samples, n_features = data.shape[0], data.shape[1]

    # normalize across samples
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # cov matrix
    cov_matrix = np.cov(data, rowvar=False)  # (n_features, n_features)

    # eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # find the k largest eigenvalues
    idx = np.argsort(-eigenvalues)[:k]  # np.argsort returns smallest to largest
    PC = eigenvectors[:, idx]  # (n_features, k)

    for i in range(k):
        evector = PC[:, i]
        for m in evector:
            if abs(m) > 1e-10 and m < 0:
                PC[:, i] = -1 * evector
                break
    
    return PC
                




