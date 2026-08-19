def kv_cache_memory(batch_size: int, seq_len: int, embed_dim: int, n_heads: int, n_layers: int, bytes_per_elem: int) -> list:
    """
    Return [mha_bytes, linear_bytes, mixed_bytes] for the three architectures.
    Mixed uses a 3:1 ratio of linear-attention layers to MHA layers.
    """
    #mha bytes
    d_head = embed_dim // n_heads
    mha_bytes = (2 * batch_size * seq_len * embed_dim * bytes_per_elem) * n_layers
    #linear bytes
    linear_bytes = (batch_size * embed_dim * d_head * bytes_per_elem) * n_layers
    #mixed bytes
    n_linear_layer = int(n_layers // 4 * 3)
    n_mha_layer = int(n_layers // 4)
    mixed_bytes = (2 * batch_size * seq_len * embed_dim * bytes_per_elem) * n_mha_layer + (batch_size * embed_dim * d_head * bytes_per_elem) * n_linear_layer
    return [mha_bytes, linear_bytes, mixed_bytes] 
