import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A = np.asarray(A, dtype=float)

    # One-sided Jacobi: pick a rotation V that makes the columns of A orthogonal.
    G = A.T @ A                       # Gram matrix
    a, b, d = G[0, 0], G[0, 1], G[1, 1]

    theta = 0.5 * np.arctan2(2.0 * b, a - d)   # angle that zeroes the off-diagonal of V^T G V
    c, s = np.cos(theta), np.sin(theta)
    V = np.array([[c, -s],
                  [s,  c]])

    B = A @ V                         # columns are now orthogonal (exactly, for 2x2)
    S = np.linalg.norm(B, axis=0)     # singular values = column norms

    # Sort descending
    order = np.argsort(-S)
    S = S[order]
    B = B[:, order]
    V = V[:, order]

    # Build U = B / S, handling (near-)zero singular values
    eps = 1e-12
    U = np.zeros((2, 2))
    if S[0] > eps:
        U[:, 0] = B[:, 0] / S[0]
        if S[1] > eps:
            U[:, 1] = B[:, 1] / S[1]
        else:
            U[:, 1] = np.array([-U[1, 0], U[0, 0]])   # any orthogonal complement
    else:
        U = np.eye(2)                                  # A is (numerically) zero

    return U, S, V.T