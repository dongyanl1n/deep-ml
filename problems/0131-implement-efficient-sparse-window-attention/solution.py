import numpy as np
def softmax(matrix):
    rowmax = np.max(matrix, axis=1, keepdims=True)
    numerator = np.exp(matrix - rowmax)
    denominator = np.sum(np.exp(matrix - rowmax), axis=1, keepdims=True)
    return numerator / denominator

def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    # Your code here
    seq_len, d_k = Q.shape
    _, d_v = V.shape
    output = np.zeros((seq_len, d_v))
    for i in range(seq_len):
        Q_i = Q[i:i+1, :]
        K_window = K[max(0, i-window_size):min(seq_len-1, i+window_size)+1, :]
        attn_score = softmax(np.matmul(Q_i, K_window.T) / np.sqrt(d_k))
        V_window = V[max(0, i-window_size):min(seq_len-1, i+window_size)+1, :]

        output[i, :] = np.matmul(attn_score, V_window)
    return output