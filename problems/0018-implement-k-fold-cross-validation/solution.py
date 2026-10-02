import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    base_fold_size = n_samples // k
    extra = n_samples % k
    fold_sizes = base_fold_size * np.ones(k, dtype=np.int_)
    fold_sizes[0] += extra

    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)  # in place
    
    folds = []
    start = 0
    for i_fold in range(k):
        end = start + fold_sizes[i_fold]
        # print(start, end)
        test_indices = indices[start:end].tolist()
        train_indices = indices[:start].tolist() + indices[end:].tolist()
        folds.append((train_indices, test_indices))

        # update start
        start = end
    return folds