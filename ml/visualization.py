import numpy as np

from .kneighbors import kneighbors_classify_matrix
from .perceptron import perceptron_vectorized


def kneighbors_boundary(X, y, k, distance_function, h=0.05):
    """
    Calculates and return the matrices with the kneighbors decision boundary.

    :param X: matrix of inputs.
    :param y: vectour of outputs.
    :param k: number of neighbors
    :param distance: distance function between two vectors.
    :param h: step for the decision boundary matrix.

    :return: three matrices needed to plot the decision boundary for the
             perceptron algorithm.
    """

    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Y = np.c_[xx.ravel(), yy.ravel()]
    Z = kneighbors_classify_matrix(X, y, Y, k, distance_function)
    Z = Z.reshape(xx.shape)

    return xx, yy, Z


def perceptron_boundary(X, y, h=0.01):
    """
    Calculates and return the matrices with the perceptron decision boundary.

    :param X: matrix of inputs.
    :param y: vectour of outputs.
    :param h: step for the decision boundary matrix.

    :return: three matrices needed to plot the decision boundary for the
             perceptron algorithm.
    """

    X_b = np.c_[np.ones(X.shape[0]), X]
    w = perceptron_vectorized(X_b, y)

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                         np.arange(y_min, y_max, h))
    Z = np.sign(np.c_[np.ones_like(xx.ravel()), 
                      xx.ravel(), 
                      yy.ravel()].dot(w)).astype(np.int)
    Z = Z.reshape(xx.shape)

    return xx, yy, Z

