import numpy as np


def perceptron(X, y, max_epoch=100):
    """
    Perceptron algorithm implementation.

    :param X: matrix of inputs.
    :param y: vector of outputs.
    :param max_epoch: maximum number of epochs for the algorithm to converge.
    """

    w = np.zeros(X.shape[1])  # Initial weights set to zero

    for _ in range(max_epoch):
        for i in range(X.shape[0]):
            xi = X[i]
            yi = y[i]
            if xi.dot(w) * yi <= 0:
                w = w + xi * yi

    return w


def perceptron_vectorized(X, y, max_epoch=100):
    """
    Perceptron algorithm implementation. Vectorized version.

    :param X: matrix of inputs.
    :param y: vector of outputs.
    :param max_epoch: maximum number of epochs for the algorithm to converge.
    """

    w = np.zeros(X.shape[1])  # Initial weights set to zero

    for _ in range(max_epoch):
        update_indices = X.dot(w) * y <= 0
        
        if np.any(update_indices):
            w += np.sum(X[update_indices] * y[update_indices].reshape(-1, 1),
                        axis=0)
        else:
            # The algorithm converges
            break

    return w.reshape(X.shape[1])

