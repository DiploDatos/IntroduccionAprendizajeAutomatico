import numpy as np


def linear_least_squares(X, y, lamda=0):
    """
    Function to calculate linear regression using the least squares method. It
    uses the Moore-Penrose pseudo-inverse of a matrix to avoid non
    invertibility.

    :param X: Matrix of inputs, where each row represents an instance and each
              column represents a feature.
    :param y: Array of target output (i.e. labels).
    :param lamda: Lambda value used for regularization.
    """

    if lamda > 0:
        reg = lamda * np.eye(X.shape[1])
        reg[0, 0] = 0
    else:
        reg = 0

    return np.linalg.pinv(X.T.dot(X) + reg).dot(X.T.dot(y))

