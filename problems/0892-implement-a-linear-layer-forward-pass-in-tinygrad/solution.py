from tinygrad import Tensor

def linear_forward(x: Tensor, W: Tensor, b: Tensor) -> Tensor:
    # TODO: implement y = x W^T + b using tinygrad ops
    # pass
    return x @ W.T + b
