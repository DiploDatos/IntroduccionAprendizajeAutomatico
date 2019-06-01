import numpy as np
import matplotlib.pyplot as plt
import itertools

from .kneighbors import kneighbors_classify_matrix
from .perceptron import perceptron_vectorized


def decision_boundary(X, w, h=0.01):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                         np.arange(y_min, y_max, h))
    Z = (np.c_[np.ones_like(xx.ravel()), 
               xx.ravel(), yy.ravel()].dot(w) >= 0.5).astype(np.int)
    Z = Z.reshape(xx.shape)

    return xx, yy, Z


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


def classifier_boundary(X, model, featurizer=None, h=0.01):
    """
    Calculates and return the matrices with the given classifier decision boundary.

    :param X: matrix of inputs.
    :param model: classifier with scikit-learn api (i.e. has a `.predict` method)
    :param featurizer: if given transform the features of the meshgrid
    :param h: step for the decision boundary matrix.

    :return: three matrices needed to plot the decision boundary for the
             perceptron algorithm.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z_features = np.c_[xx.ravel(), yy.ravel()]

    if featurizer is not None:
        Z_features = featurizer.transform(Z_features)

    Z = model.predict(Z_features)
    Z = Z.reshape(xx.shape)

    return xx, yy, Z


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta correcta')
    plt.xlabel('Etiqueta predicha')


def plot_learning_curve(train_sizes, train_scores, validation_scores,
                        title="Curva de aprendizaje"):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    
    plt.title(title)
    plt.xlabel("Cantidad de datos de entrenamiento")
    plt.ylabel("Exactitud")
    plt.ylim(0.0, 1.1)
    plt.grid()
    
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.5, color="lightcoral")
    plt.fill_between(train_sizes,
                     validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std,
                     alpha=0.5, color="skyblue")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="tomato",
             label="Datos de entrenamiento")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="dodgerblue",
             label="Datos de validación")
    plt.legend(loc="lower right")
