import numpy as np

def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """
    Simplified Gated DeltaNet linear attention forward pass.

    Args:
        q, k, v: lists of shape (T, d)
        a, b: lists of shape (T,) -- pre-activation gate logits
        g: list of shape (T, d) -- output gate logits
        A_log: float -- learned log time-scale
        rms_weight: list of shape (d,) -- RMSNorm scale
        eps: float -- numerical stability constant

    Returns:
        Nested list of shape (T, d), rounded to 4 decimals.
    """
    # Your code here
    def softplus(z):
        return np.log(1 + np.exp(z))
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    q = np.array(q)
    k = np.array(k) 
    v = np.array(v)
    g = np.array(g)
    a = np.array(a)
    b = np.array(b)
    rms_weight = np.array(rms_weight)
    T,d = q.shape
    # L2-normalize q and k along the feature dimension
    q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True)  + eps) # (T,d)
    k_norm = k / (np.linalg.norm(k, axis=-1, keepdims=True)  + eps) # (T,d)
    # Compute decay gate per timestep
    alpha = np.exp(-softplus(a)) * np.exp(A_log)  # (T,)
    # Compute update gate per timestep
    beta = sigmoid(b)  # (T,)
    # Memory update
    S = np.zeros((d,d))
    y = []
    for t in range(T):
        # Decay
        S = alpha[t] * S
        # Prediction error
        delta = (v[t] - S.T @ k_norm[t]) * beta[t] # (d,)
        # Memory update
        S = S + np.outer(k_norm[t], delta)  # d,d
        # Output
        o_t = S.T @ q_norm[t]
        o_norm_t = (o_t / np.sqrt(np.mean(o_t**2) + eps)) * rms_weight
        y_t = o_norm_t * (g[t] * sigmoid(g[t]))
        y.append(np.round(y_t, 4).tolist())
    return y


