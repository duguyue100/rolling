"""
Some drawing functions for producing right figures

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def draw_rf_boxplot(data, filename):
    """Draw box plot for output rolling force.

    Parameter
    ---------
    data : 1-d vector
        rolling force vector
    filename : string
        filename you want to save

    Returns
    -------
    Save figure as png and eps format
    """
    plt.figure()
    plt.boxplot(data, color='k')
    plt.ylabel("Rolling Force")
    plt.savefig("../results/" + filename + ".png", dpi=200)
    plt.savefig("../results/" + filename + ".eps", dpi=200)


def draw_epochs_cost(cost, filename):
    """Draw figure of epochs vs cost for both training and testing set.

    Parameters
    ----------
    cost : array
        array of training set cost and testing cost
        1st row : train cost
        2nd row : test cost
    filename : string
        filename you want to save

    Returns
    -------
    save figures in png and eps format
    """
    num_epochs = cost.shape[1]
    x = np.array(xrange(num_epochs)) + 1

    train_min = np.min(cost[0, :])
    test_min = np.min(cost[1, :])
    train_min_x = np.argmin(cost[0, :])
    test_min_x = np.argmin(cost[0, :])

    plt.figure()
    train_cost, = plt.plot(
        x, cost[0, :], linestyle='-', color='k',
        linewidth=2, label="train cost")
    test_cost, = plt.plot(
        x, cost[1, :], linestyle='--', color='k',
        linewidth=2, label="test cost")
    plt.plot(train_min_x, train_min, marker="*", markersize=12, color='k')
    plt.plot(test_min_x, test_min, marker="D", markersize=8, color='k')
    plt.ylabel("cost")
    plt.xlabel("epochs")
    plt.legend(handles=[train_cost, test_cost])
    plt.savefig("../results/" + filename + ".png", dpi=200)
    plt.savefig("../results/" + filename + ".eps", dpi=200)

    return


def draw_target_predicted(target, predicted, filename):
    """Draw figure of target output vs predicted output.

    Parameters
    ----------
    target : array
        1-d vector of target output
    predicted : array
        1-d vector of predicted output
    filename : string
        filename you want to save
    filename : string
        filename you want to save

    Returns
    -------
    save figure in png and eps format
    """
    num_samples = target.shape[0]
    data = np.zeros((1, 2))
    for i in xrange(num_samples):
        if np.abs(target[i, 0] - predicted[i, 0]) < 100:
            tmp = np.array([target[i, 0], predicted[i, 0]])
            data = np.vstack((data, tmp))

    data = data[1:, :]

    coeff = np.polyfit(data[:, 0], data[:, 1], 1)
    fit_fun = np.poly1d(coeff)
    x = np.array(xrange(1000, 5000))
    y = fit_fun(x)

    plt.figure()
    plt.plot(data[:, 0], data[:, 1], '.', color='k')
    plt.plot(x, y, linestyle='-', color='k', linewidth=2)
    plt.axis([1500, 4500, 1500, 4500])
    plt.xlabel("Target outputs")
    plt.ylabel("Predicted outputs")
    plt.savefig("../results/" + filename + ".png", dpi=200)
    plt.savefig("../results/" + filename + ".eps", dpi=200)

    return


def draw_cost_algorithms(cost, filename):
    """Draw figure of cost vs training algorithm.

    Parameters
    ----------
    cost : array
        (k methods) x (number of epochs) matrix in order
        1st row: SGD
        2nd row: momentum SGD
        3rd row: AdaGrad
        4th row: RMSprop
    filename : string
        filename you want to save

    Returns
    -------
    Save figure in png and eps format
    """
    num_epochs = cost.shape[1]
    x = np.array(xrange(num_epochs)) + 1

    plt.figure()
    sgd, = plt.plot(x, cost[0, :], linestyle='-',
                    linewidth=2, color='k', label="SGD")  # SGD
    momentum, = plt.plot(x, cost[1, :], linestyle='--', linewidth=2,
                         color='k', label="Momentum SGD")  # momentum SGD
    adagrad, = plt.plot(x, cost[2, :], linestyle='-.',
                        linewidth=2, color='k', label="AdaGrad")  # AdaGrad
    rmsprop, = plt.plot(x, cost[3, :], linestyle=':',
                        linewidth=2, color='k', label="RMSprop")  # RMSprop
    plt.ylabel("cost")
    plt.xlabel("epochs")
    plt.legend(handles=[sgd, momentum, adagrad, rmsprop])
    plt.savefig("../results/" + filename + ".png", dpi=200)
    plt.savefig("../results/" + filename + ".eps", dpi=200)

    return


def draw_neurons_layers_cost(cost, filename):
    """Draw figures number of neurons vs cost.

    All for feedforward layer.

    Parameters
    ----------
    cost : array
        (k layers) x (max number of neurons) matrix in order
        1st row : 1 hidden layer
        2nd row : 2 hidden layers
        3rd row : 3 hidden layers
        4th row : 4 hidden layers
        5th row : 5 hidden layers
    filename : string
        filename you want to save

    Returns
    -------
    save figure in png and eps format
    """
    num_neurons = cost.shape[1]
    x = np.array(xrange(num_neurons)) + 1

    plt.figure()
    lin_1, = plt.plot(x, cost[0, :], linestyle='-', marker='v', color='k',
                      linewidth=2, labels='1 hidden layer')  # 1 hidden layer
    lin_2, = plt.plot(x, cost[1, :], linestyle='-', marker='s', color='k',
                      linewidth=2, labels='2 hidden layers')  # 2 hidden layers
    lin_3, = plt.plot(x, cost[2, :], linestyle='-', marker='p', color='k',
                      linewidth=2, labels='3 hidden layers')  # 3 hidden layers
    lin_4, = plt.plot(x, cost[3, :], linestyle='-', marker='h', color='k',
                      linewidth=2, labels='4 hidden layers')  # 4 hidden layers
    lin_5, = plt.plot(x, cost[4, :], linestyle='-', marker='*', color='k',
                      linewidth=2, labels='5 hidden layers')  # 5 hidden layers
    plt.ylabel("cost")
    plt.xlabel("number of neurons")
    plt.legend(handles=[lin_1, lin_2, lin_3, lin_4, lin_5])
    plt.savefig("../results/" + filename + ".png", dpi=200)
    plt.savefig("../results/" + filename + ".eps", dpi=200)

    return


def draw_cost_dropout(cost, filename):
    """Draw figure of number of epochs vs cost.

    Difference is with or without dropout

    Parameters
    ----------
    cost : arrray
        2 x number of epochs matrix in order
        1st row : without dropout
        2nd row : with dropout
    filename : string
        filename you want to save
    """
    num_eopchs = cost.shape[1]
    x = np.array(xrange(num_eopchs)) + 1

    plt.figure()
    dropout, = plt.plot(x, cost[0, :], linestyle='-', color='k',
                        linewidth=2, labels="Dropout")  # without dropout
    l2, = plt.plot(x, cost[1, :], linestyle='--', color='k',
                   linewidth=2, labels="L2")     # with dropout
    plt.ylabel("cost")
    plt.xlabel("eopchs")
    plt.legend(handles=[dropout, l2])
    plt.savefig("../results/" + filename + ".png", dpi=200)
    plt.savefig("../results/" + filename + ".eps", dpi=200)

    return
