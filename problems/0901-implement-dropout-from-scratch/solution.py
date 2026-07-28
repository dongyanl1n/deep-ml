import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    # pass
    if training:
        mask = (torch.rand(*x.shape) >= p)  # 0 or 1
        x = x*mask*(1/(1-p))
        return x
    else: 
        return x
