import numpy as np

def softmax(z):
    # z: B, L, L
    z_max = np.max(z, axis=-1, keepdims=True)
    e_z = np.exp(z - z_max)
    return e_z / np.sum(e_z, axis=-1, keepdims=True)

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    head_dim = Q.shape[-1] // num_heads
    B, L = Q.shape[0], Q.shape[1]
    num_heads_per_kv_group = num_heads // num_kv_heads
    Q_heads = np.split(Q, num_heads, axis=-1)  # len: num_heads
    K_heads = np.split(K, num_kv_heads, axis=-1)  # len: num_kv_heads
    V_heads = np.split(V, num_kv_heads, axis=-1)  # len: num_kv_heads
    output = np.zeros_like(Q)
    for i_kv_head in range(num_kv_heads):  # for each group
        start = i_kv_head * num_heads_per_kv_group
        Q_h_group = Q_heads[start : start + num_heads_per_kv_group]
        K_h = K_heads[i_kv_head]  # B, L, head_dim
        V_h = V_heads[i_kv_head]  # B, L, head_dim
        for i_head in range(num_heads_per_kv_group):
            Q_h = Q_h_group[i_head]  # B, L, head_dim
            attn_score = softmax(np.matmul(Q_h, K_h.transpose(0, 2, 1)) / np.sqrt(head_dim))  # B, L, L
            out_h = np.matmul(attn_score, V_h)  # B, L, head_dim
            head_idx = start + i_head
            output[:, :, head_idx * head_dim : (head_idx + 1) * head_dim] = out_h
    return output