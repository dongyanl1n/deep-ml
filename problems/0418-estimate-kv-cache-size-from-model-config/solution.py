def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    # Your code here
    if 'num_kv_heads' not in model_config:
        model_config['num_kv_heads'] = model_config['num_attention_heads']
    
    head_dim = model_config['hidden_size'] / model_config['num_attention_heads']


    output_dict = {}
    output_dict['kv_cache_elements'] = 2 * model_config['num_layers'] * model_config['num_kv_heads'] *head_dim * batch_size * seq_len
    output_dict['kv_cache_size_bytes'] = model_config['dtype_bytes'] * output_dict['kv_cache_elements']
    output_dict['kv_cache_size_mb'] = round(output_dict['kv_cache_size_bytes'] / 1024**2, 4)
    output_dict['per_layer_size_mb'] = round(output_dict['kv_cache_size_mb'] / model_config['num_layers'], 4)
    output_dict['per_token_size_kb'] = round(2 * model_config['num_layers'] * model_config['num_kv_heads'] * head_dim * model_config['dtype_bytes'] / 1024, 4)
    
    return output_dict