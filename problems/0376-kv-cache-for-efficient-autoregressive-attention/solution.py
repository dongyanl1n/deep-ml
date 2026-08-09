import numpy as np

def softmax(z):
    # assume z.shape is (L,)
    out = np.exp(z-np.max(z)) / np.sum(np.exp(z-np.max(z)))
    return out


def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    # pass
    # Project the new token embedding into query, key, and value vectors using the provided weight matrices
    d_model, d_k = W_Q.shape
    _, d_v = W_V.shape

    Q, K, V = W_Q.T @ x_new,  W_K.T @ x_new,  W_V.T @ x_new # (d_k, ) or (d_v, )

    # Append the new key and value to the existing cache (or create a new cache if none exists)
    # assume shape convention: L x d_k
    if cache is None:  # first token
        K_cache = K.reshape((1, d_k))
        V_cache = V.reshape((1, d_v))
    else:
        K_cache, V_cache = cache
        K_cache = np.concatenate([K_cache, K.reshape((1, d_k))], axis=0)  # (L, d_k)
        V_cache = np.concatenate([V_cache, V.reshape((1, d_v))], axis=0)  # (L, d_v)
    cache = (K_cache, V_cache)
    
    # Compute scaled dot-product attention between the new query and all cached keys
    # attn_score = softmax(K_cache @ Q) / np.sqrt(d_k)  # (L, )
    attn_score = softmax(K_cache @ Q / np.sqrt(d_k))  # for the love of god, scale BEFORE softmax
    output = attn_score @ V_cache  # (d_v, )

    # Return the attention output vector and the updated cache
    return output, cache

