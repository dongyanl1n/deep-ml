import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    # pe = torch.zeros((seq_len, d_model))
    # for i in range(0, d_model-1, 2):
    #     for pos in range(seq_len):
    #         pe[pos, 2*i] = math.sin(pos * math.exp(-math.log(10000) *  2*i / d_model))
    #         pe[pos, 2*i+1] = math.cos(pos * math.exp(-math.log(10000) *  2*i / d_model))
    
    # Compute the positional encodings once in log space.
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)  # shape (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )  # (d_model // 2,)
    pe[:, 0::2] = torch.sin(position * div_term)  # * for outer, @ for inner??
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
