import numpy as np


def kneighbors(X, x, k, distance_function):
    """
    Returns the `k` neighbors to the point `x` to the points of `X` according
    to the `distance_function` given by the user.

    :param X: Matrix of points to make comparison.
    :param x: Point to look the k neighbors.
    :param k: Number of neighbors.
    :param distance_function: Distance function between to vectors.
    :return: k nearest neighbors of point `x`.
    """

    distances = np.fromiter((distance_function(x_, x) for x_ in X),
                             dtype=np.float)
    return distances.argsort()[:k]


def kneighbors_classify_point(X, y, x, k, distance_function):
    """
    Classifies the point `x` w.r.t. the training set given by points `X` and
    labels `y`.

    :param X: Matrix of points to make comparison.
    :param y: Array of labels.
    :param x: Point to classify.
    :return: The classification of `x` using the k nearest neighbors approach.
    """

    neighbors = kneighbors(X, x, k, distance_function)
    counts = np.bincount(y[neighbors].astype(np.int))
    return counts.argmax()


def kneighbors_classify_matrix(X, y, Y, k, distance_function):
    """
    Classifies each point in the Matrix `Y` w.r.t the training set given by
    points `X` and labels `y`.

    :param X: Matrix of points to make comparison.
    :param y: Array of labels.
    :param Y: Matrix to classify.
    :param k: Number of neighbors.
    :param distance_function: Distance function between to vectors.
    :return: The classification of `x` using the k nearest neighbors approach.
    """

    output = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        output[i] = kneighbors_classify_point(X, y, Y[i], k, distance_function)

    return output

