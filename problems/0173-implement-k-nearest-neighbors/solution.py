import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    # pass
    def distance(point_a, point_b):
        # euclidean distance
        sum_of_squares = 0
        for a, b in zip(point_a, point_b):
            sum_of_squares += (a-b)**2
        return np.sqrt(sum_of_squares)

    distances = np.zeros(len(points))
    for i, p in enumerate(points):
        distances[i] = distance(p, query_point)
    indices = np.argsort(distances)[:k]  # smallest to biggest distance

    return [points[i] for i in indices]
