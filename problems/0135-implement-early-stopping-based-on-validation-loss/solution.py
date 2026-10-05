from typing import Tuple
import numpy as np
def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:

    wait = 0
    best_epoch = 0
    # best_val_loss = 0
    best_val_loss = float('inf')  # this is key!!

    for i_epoch, loss in enumerate(val_losses):
        if loss < best_val_loss - min_delta:  # count as improvement
            best_epoch = i_epoch
            best_val_loss = loss
            wait = 0  # reset
        else:  # no improvement
            wait += 1
        
        if wait >= patience:
            return (i_epoch, best_epoch)
        
    return (len(val_losses)-1, best_epoch)



