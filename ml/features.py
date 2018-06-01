import functools
import itertools
import numpy as np

def polynomial_features(X, degree):
    """
    Returns the polynomial features of the given input matrix X.

    :param X: Numpy matrix with the inputs in each row and the features in each
              column.
    :param degree: Degree of the polynomial features.
    """
    if X.ndim == 1:
        # Treat the numpy arrays as mathematical column vectors
        X = X.reshape(-1, 1)
    features = [np.ones(X.shape[0])]
    for degree in range(1, degree + 1):
        for items in itertools.combinations_with_replacement(X.T, degree):
            features.append(functools.reduce(lambda x, y: x * y, items))
    return np.asarray(features).T
