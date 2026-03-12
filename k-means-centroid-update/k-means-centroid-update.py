def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    n = len(points[0])
    centroids = [[0.] * n for i in range(k)]

    cnt = [0] * k

    for point, assign in zip(points, assignments):
        for i in range(n):
            centroids[assign][i] += point[i]
        cnt[assign] += 1

    for i in range(k):
        if cnt[i] == 0:
            centroids[i] = [0.] * n
        else:
            centroids[i] = [centroids[i][j] / cnt[i] for j in range(n)]
    return centroids