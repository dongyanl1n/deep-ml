import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    y_log_p = []
    for i in range(len(predicted_probs)):
        true_labels_i, predicted_probs_i = true_labels[i], predicted_probs[i]
        y_log_p_i = []
        for y_i, p_i in zip(true_labels[i], predicted_probs[i]):
            y_log_p_i.append(y_i * np.log(p_i + epsilon))
        y_log_p.append(-np.sum(y_log_p_i)) # sum over class, negate
    return np.mean(np.asarray(y_log_p))  # mean over samples