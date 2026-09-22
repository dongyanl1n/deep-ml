import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x):
	rowmax = np.max(x, axis=1, keepdims=True)
	expo = np.exp(x - rowmax)
	sum_expo = np.sum(expo, axis=1, keepdims=True)
	return expo / sum_expo

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	score = np.matmul(Q, K.T) / np.sqrt(K.shape[1])
	masked_score = score + mask
	masked_attn_score = softmax(masked_score)
	return np.matmul(masked_attn_score, V)