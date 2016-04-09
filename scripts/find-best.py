"""Find the best result from experiments.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import argparse
import cPickle as pickle
import numpy as np

import rolling.dataset as ds


def search(results_path, network_type, batch_size, num_epochs):
    """Find all available results available in results destination.

    Parameters
    ----------
    results_path : string
        destination of experiment results
    network_type : string
        feedforward or RNN
    batch_size : string
        batch size for each mini-batch samples
    num_epochs : string
        total number of training epochs

    Returns
    -------
    result_lit : List
        a list of relevant result files
    """
    all_layers = xrange(1, 6)
    all_neurons = xrange(10, 305, 5)
    all_methods = ["sgd", "momentum", "adagrad", "rmsprop"]
    all_regularization = ["dropout", "l2"]
    result_list = []

    for n_layers in all_layers:
        for n_neurons in all_neurons:
            for method in all_methods:
                for regular in all_regularization:
                    exp_id = ds.create_exp_id(network_type, n_layers,
                                              n_neurons, batch_size,
                                              num_epochs, method, regular)
                    file_path = os.path.join(results_path, exp_id+".pkl")
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            result_list.append(pickle.load(f))

    return result_list


def analysis(results_path, network_type, batch_size, num_epochs):
    """Find the best result from rolling experiment results.

    Parameters
    ----------
    results_path : string
        destination of the results
    network_type : string
        feedforward or RNN
    batch_size : string
        batch size for each mini-batch samples
    num_epochs : string
        total number of training epochs

    Returns
    -------
    best_id : string
        Identifier of best result
    """
    result_list = search(results_path, network_type, batch_size, num_epochs)

    best_result = result_list[0]
    worst_result = result_list[0]

    for i in xrange(1, len(result_list)):
        temp_result = result_list[i]

        if np.min(temp_result['cost'][1, :]) < np.min(
                best_result['cost'][1, :]):
            best_result = temp_result
        elif np.min(temp_result['cost'][1, :]) > np.min(
                best_result['cost'][1, :]):
            worst_result = temp_result

    best_tr_cost = np.min(best_result['cost'][0, :])
    best_tr_epoch = np.argmin(best_result['cost'][0, :])
    best_te_cost = np.min(best_result['cost'][1, :])
    best_te_epoch = np.argmin(best_result['cost'][1, :])

    worst_tr_cost = np.min(worst_result['cost'][0, :])
    worst_tr_epoch = np.argmin(worst_result['cost'][0, :])
    worst_te_cost = np.min(worst_result['cost'][1, :])
    worst_te_epoch = np.argmin(worst_result['cost'][1, :])

    print "-------------------------------------------------------------------"
    print "-------------------------------------------------------------------"
    print "Best result experiment ID: %s" % (best_result['exp_id'])
    print "Best result training cost %f in epoch %d" % (best_tr_cost,
                                                        best_tr_epoch)
    print "Best result testing cost %f in epoch %d" % (best_te_cost,
                                                       best_te_epoch)
    print "-------------------------------------------------------------------"
    print "-------------------------------------------------------------------"
    print "Worst result experiment ID: %s" % (worst_result['exp_id'])
    print "Worst result training cost %f in epoch %d" % (worst_tr_cost,
                                                         worst_tr_epoch)
    print "Worst result testing cost %f in epoch %d" % (worst_te_cost,
                                                        worst_te_epoch)
    print "-------------------------------------------------------------------"
    print "-------------------------------------------------------------------"

    return

parser = argparse.ArgumentParser(
    description="Find Best Result From Rolling Experiments")

parser.add_argument("--results-path", type=str,
                    default="/Users/dgyHome/Downloads/results",
                    help="Destination of result files.")
parser.add_argument("--network-type", type=str, default="feedforward",
                    help="Type of network: feedforward or RNN.")
parser.add_argument("--batch-size", type=str, default="200",
                    help="Batch size of each mini-batch samples.")
parser.add_argument("--num-epochs", type=str, default="1000",
                    help="Total training epochs for training.")

args = parser.parse_args()

analysis(**vars(args))
