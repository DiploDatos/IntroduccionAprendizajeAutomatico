import numpy as np

from sklearn.datasets import make_classification, make_moons


def create_lineal_data(slope=1, bias=0, spread=0.25, data_size=50):
    """
    Helper function to create lineal data.

    :param slope: slope of the lineal function.
    :param bias: bias of the lineal function.
    :param spread: spread of the normal distribution.
    :param data_size: number of samples to generate.

    :return x, y: data and labels
    """
    x = np.linspace(0, 1, data_size)
    y = x * slope + bias + np.random.normal(scale=spread, size=x.shape)

    return x, y


def sin_function(x):
    """
    Returns the sin function complete on interval [0, 1] by scaling the input
    by a 2*pi factor.

    :param x: input data vector.

    :returns: sin function of the data vector on interval [0, 1]
    """

    return np.sin(2 * np.pi * x)


def create_sinusoidal_data(spread=0.25, data_size=50):
    """
    Creates a sinusoidal dataset.

    :param spread: spread of the normal distribution.
    :param data_size: number of samples to generate.

    :return x, y: data and labels
    """
    x = np.linspace(0, 1, data_size)
    y = sin_function(x) + np.random.normal(scale=spread, size=x.shape)

    return x, y


def create_classification_data(data_kind, negative_label=-1, data_size=100):
    """
    Creates a classification toy dataset using the scikit learn helper 
    functions.

    :param data_kind: string in {'lineal', 'nonlineal'}
    :negative_label: integer used for the negative label.
    :param data_size: number of samples.

    :return x, y: data and labels
    """

    assert data_kind in {'lineal', 'nonlineal'},\
            "The data kind parameter should be either 'lineal' or 'nonlineal'"

    if data_kind == 'lineal':
        X, y = make_classification(n_samples=data_size, n_features=2,
                                   n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, class_sep=1.5,
                                   random_state=5)
    else:
        X, y = make_moons(noise=0.1, random_state=5)

    y[y == 0] = negative_label

    return X, y

