import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q, K, V = np.matmul(X, W_q), np.matmul(X, W_k), np.matmul(X, W_v)
    return Q, K, V

def softmax(matrix):
    row_max = np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(matrix - row_max)
    row_sum_exp = np.sum(exp_matrix, axis=1, keepdims=True)
    return exp_matrix / row_sum_exp

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    return np.matmul(softmax(np.matmul(Q, K.T) / np.sqrt(Q.shape[-1])) , V)


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    output_matrix = []
    d_k = int(Q.shape[-1] / n_heads)
    for i_head in range(n_heads):
        Q_head = Q[:, i_head*d_k : i_head*d_k+d_k]
        K_head = K[:, i_head*d_k : i_head*d_k+d_k]
        V_head = V[:, i_head*d_k : i_head*d_k+d_k]
        output_matrix.append((self_attention(Q_head, K_head, V_head)))  # seq_len, d_k
    output_matrix = np.concatenate(output_matrix, axis=1)  # seq_len, d_model
    return output_matrix



