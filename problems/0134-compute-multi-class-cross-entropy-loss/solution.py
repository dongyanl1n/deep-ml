import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    # pass
    n_samples, n_classes = predicted_probs.shape[0], predicted_probs.shape[1]
    arr = []
    for i in range(n_samples):
        pred = predicted_probs[i]
        label = true_labels[i]
        arr.append(pred[label==1])
    arr = np.array(arr)
    return -np.mean(np.log(arr+epsilon))
        
