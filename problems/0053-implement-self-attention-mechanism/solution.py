import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def softmax(matrix):
    n_row, n_col = matrix.shape
    result = []
    for i_row in range(n_row):
        row = matrix[i_row]
        result.append( np.exp(row) / np.sum(np.exp(row)))
    return np.array(result)


def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    attn_scores = np.matmul(Q, K.T) / np.sqrt(K.shape[1])
    attn_scores = softmax(attn_scores)
    output = np.matmul(attn_scores, V)
    return output
    # pass
