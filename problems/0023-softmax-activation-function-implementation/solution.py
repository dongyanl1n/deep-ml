import math

def softmax(scores: list[float]) -> list[float]:
    listmax = max(scores)
    expo = [score - listmax for score in scores]
    numerator = [math.exp(e) for e in expo]
    denominator = sum(numerator)
    return [n / denominator for n in numerator]
    # pass