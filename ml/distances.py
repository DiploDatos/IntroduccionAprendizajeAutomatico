import numpy as np


def euclidean(u, v):
    """
    Calculates the euclidean distance between vectors u an v

    precondition:
        u.shape == v.shape

    :param u: vector of numbers
    :param v: vector of numbers
    :return: euclidean distance between u and v
    """

    if u.shape != v.shape:
        raise ValueError("The size of u and v differs")

    return np.sqrt(np.power(u - v, 2).sum())


def manhattan(u, v):
    """
    Calculates the manhattan distance (also known as taxicab geometry) between
    vectors u an v.

    precondition:
        u.shape == v.shape

    :param u: vector of numbers
    :param v: vector of numbers
    :return: manhattan distance between u and v
    """

    if u.shape != v.shape:
        raise ValueError("The size of u and v differs")

    return np.abs(u - v).sum()


def cosine(u, v):
    """
    Calculates the cosine distance (1 - cosine similarity) between two vectors

    precondition:
        u.shape == v.shape

    :param u: vector of numbers
    :param v: vector of numbers
    :return: cosine distance between u and v
    """

    if u.shape != v.shape:
        raise ValueError("The size of u and v differs")

    return 1 - u.dot(v) / np.sqrt(u.dot(u)*v.dot(v))
