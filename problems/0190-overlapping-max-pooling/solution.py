import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    N,C,H,W = x.shape
    # N_out = int(np.ceil((N-kernel_size) / stride) + 1)
    # C_out = int(np.ceil((C-kernel_size) / stride) + 1)
    N_out, C_out = N, C # Max pooling only applies to H and W
    H_out = int(np.ceil((H-kernel_size) / stride) + 1)
    W_out = int(np.ceil((W-kernel_size) / stride) + 1)
    # output_array = np.zeros((N_out, C_out, H_out, W_out))
    output_array = np.zeros((N_out, C_out, H_out, W_out), dtype=x.dtype)

    for i_n in range(N_out):
        for i_c in range(C_out):
            for i_h in range(H_out):
                for i_w in range(W_out):
                    # start_n = i_n * stride
                    # end_n = min(start_n+stride, N)
                    # start_c = i_c * stride
                    # end_c = min(start_c+stride, C)
                    start_h = i_h * stride
                    end_h = min(start_h+kernel_size, H)
                    start_w = i_w * stride
                    end_w = min(start_w+kernel_size, W) # Step by stride, but extend by kernel_size!

                    # window = x[start_n:end_n, start_c:end_c, start_h:end_h, start_w:end_w]
                    window = x[i_n, i_c, start_h:end_h, start_w:end_w]  # No need to pool over N and C!

                    output_array[i_n, i_c, i_h, i_w] = np.max(window)
    return output_array
