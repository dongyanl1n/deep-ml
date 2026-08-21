import numpy as np

FP8_E4M3_MAX = 448.0        # max 

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    # Your code here
    tensor = np.asarray(tensor, dtype=np.float64)
    n = tensor.shape[0]
    assert n % block_size == 0, "tensor length must be divisible by block_size"
    num_blocks = n // block_size

    blocks = tensor.reshape(num_blocks, block_size)
    amax = np.max(np.abs(blocks), axis=1)

    # avoid divide-by-zero for all-zero blocks
    scales = np.where(amax > 0, amax / FP8_E4M3_MAX, 1.0)

    scaled = blocks / scales[:, None]
    quantized = np.clip(np.round(scaled), -FP8_E4M3_MAX, FP8_E4M3_MAX)

    return quantized.reshape(n), scales



def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    # Your code here
    quantized = np.asarray(quantized, dtype=np.float64)
    n = quantized.shape[0]
    num_blocks = n // block_size
    blocks = quantized.reshape(num_blocks, block_size)
    return (blocks * scales[:, None]).reshape(n)