import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    def euclidean_distance(tup_a, tup_b):
        total_sum = 0
        for a, b in zip(tup_a, tup_b):
            total_sum += (a - b) ** 2
        return np.sqrt(total_sum)

    points_arr = np.array(points, dtype=float)
    centroids = [tuple(c) for c in initial_centroids]
    epsilon = 1e-8

    distance_matrix = np.zeros((len(points), k))
    final_centroids = [tuple(np.round(c, 4)) for c in centroids]  # make a copy of the list to avoid mutation

    for iteration in range(max_iterations):
        # Step 1: assign each point to its nearest centroid
        for i_p, point in enumerate(points):
            for i_c, cent in enumerate(centroids):
                distance_matrix[i_p, i_c] = euclidean_distance(point, cent)

        assignments = np.argmin(distance_matrix, axis=1)  # the cluster index that each point is assigned to

        # Step 2: recompute centroids as mean of assigned points
        new_centroids = []
        for i_c in range(k):
            cluster_points = points_arr[assignments == i_c]
            if len(cluster_points) > 0:
                new_centroids.append(tuple(cluster_points.mean(axis=0)))
            else:
                new_centroids.append(centroids[i_c])  # not updating the centroid

        # Step 3: check convergence
        shift = np.linalg.norm(np.array(new_centroids) - np.array(centroids))
        centroids = new_centroids
        final_centroids = [tuple(np.round(c, 4)) for c in centroids]

        if shift <= epsilon:
            break

    return final_centroids