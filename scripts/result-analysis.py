"""
This piece of code analyze and produce proper figures based on result file(s).

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import argparse
import cPickle as pickle
import numpy as np

import rolling.dataset as ds
import rolling.draw as draw


def search(results_path, network_type, num_layers,
           num_neurons, batch_size, num_epochs,
           training_method, regularization):
    """Search relevant files.

     based on input arguments and return a list of filename

    Parameters
    ----------
    results_path : string
        Destination of result files
    network_type : string
        feedforward or RNN
    num_layers : string
        either integer from 1-5 or "all"
    num_neurons : string
        either integer from 10-300 or "all"
    batch_size : string
        batch size for each mini-batch samples
    num_epochs : string
        total number of training epochs
    training method : string
        sgd, momentum, adagrad, rmsprop, all
    regularization : string
        l2, dropout or none, all

    Returns
    -------
    result_list : Dictionary
        a list of relevant result files
    """
    all_layers = xrange(1, 6)
    all_neurons = xrange(10, 305, 5)
    all_methods = ["sgd", "momentum", "adagrad", "rmsprop"]
    all_regularization = ["dropout", "l2"]
    result_list = []

    if num_layers != "all":
        all_layers = [int(num_layers)]
    if num_neurons != "all":
        all_neurons = [int(num_neurons)]
    if training_method != "all":
        all_methods = [training_method]
    if regularization != "all":
        all_regularization = [regularization]

    for n_layers in all_layers:
        for n_neurons in all_neurons:
            for method in all_methods:
                for regular in all_regularization:
                    exp_id = ds.create_exp_id(network_type, n_layers,
                                              n_neurons, batch_size,
                                              num_epochs, method, regular)
                    file_path = os.path.join(results_path, exp_id+".pkl")
                    with open(file_path, 'r') as f:
                        result_list.append(pickle.load(f))

    return result_list


def analysis(results_path, network_type, num_layers,
             num_neurons, batch_size, num_epochs,
             training_method, regularization, mode):
    """An Analysis result from given arguments.

    Parameters
    ----------
    results_path : string
        Destination of result files
    network_type : string
        feedforward or RNN
    num_layers : string
        either integer from 1-5 or "all"
    num_neurons : string
        either integer from 10-300 or "all"
    batch_size : string
        batch size for each mini-batch samples
    num_epochs : string
        total number of training epochs
    training method : string
        sgd, momentum, adagrad, rmsprop, all
    regularization : string
        l2, dropout or none, all
    mode : string
        output mode: targets-predicted, epochs-cost, cost-algorithm,
        neurons-cost, cost-regular
    """
    if mode == "targets-predicted":
        assert num_layers != "all", "num-layers should be 1-5 \
                                     in targets-predicted mode"
        assert num_neurons != "all", "num-neurons should be 10-300 \
                                      in targets-predicted mode"
        assert training_method != "all", "training-method shouldn't \
                                          be all in targets-predicted mode"
        assert regularization != "all", "regularization shouldn't be all \
                                         in targets-predicted mode"

        results_list = search(results_path, network_type, num_layers,
                              num_neurons, batch_size, num_epochs,
                              training_method, regularization)
        result = results_list[0]
        train_fn = os.path.join(results_path, "images",
                                result["exp_id"]+"_targets-predicted-train")
        test_fn = os.path.join(results_path, "images",
                               result['exp_id']+"_targets-predicted-test")
        draw.draw_target_predicted(result['train_targets'], result[
                                   'train_predicted'],
                                   train_fn)
        draw.draw_target_predicted(result['test_targets'], result[
                                   'test_predicted'],
                                   test_fn)
    elif mode == "epochs-cost":
        assert num_layers != "all", "num-layers should be 1-5 \
                                     in epochs-cost mode"
        assert num_neurons != "all", "num-neurons should be 10-300 \
                                      in epochs-cost mode"
        assert training_method != "all", "training-method shouldn't be \
                                          all in epochs-cost mode"
        assert regularization != "all", "regularization shouldn't be \
                                         all in epochs-cost mode"
        results_list = search(results_path, network_type, num_layers,
                              num_neurons, batch_size, num_epochs,
                              training_method, regularization)
        result = results_list[0]
        fn = os.path.join(results_path, "images",
                          result['exp_id']+"_epochs-cost")
        draw.draw_epochs_cost(result['cost'], fn)
    elif mode == "cost-algorithm":
        assert num_layers != "all", "num-layers should be 1-5 \
                                     in cost-algorithm mode"
        assert num_neurons != "all", "num-neurons should be 10-300 \
                                      in cost-algorithm mode"
        assert training_method == "all", "training-method should be all \
                                          in cost-algorithm mode"
        assert regularization != "all", "regularization shouldn't be all \
                                         in cost-algorithm mode"
        results_list = search(results_path, network_type, num_layers,
                              num_neurons, batch_size, num_epochs,
                              training_method, regularization)
        cost_arr = results_list[0]['cost'][1, :]
        for k in xrange(1, len(results_list)):
            cost_arr = np.vstack((cost_arr, results_list[k]['cost'][1, :]))

        fn = ds.create_exp_id(network_type, num_layers,
                              num_neurons, batch_size,
                              num_epochs, "all",
                              regularization)
        fn = os.path.join(results_path, "images", fn+"_cost-algorithm")
        draw.draw_cost_algorithms(cost_arr, fn)
    elif mode == "neurons-cost":
        assert num_layers == "all", "num-layers should be all \
                                     in neurons-cost mode"
        assert num_neurons == "all", "num-neurons should be all \
                                      in neurons-cost mode"
        assert training_method != "all", "training-method shouldn't be \
                                          all in neurons-cost mode"
        assert regularization != "all", "regularization shouldn't be \
                                         all in neurons-cost mode"
        results_list = search(results_path, network_type, num_layers,
                              num_neurons, batch_size, num_epochs,
                              training_method, regularization)
        cost_arr = np.zeros((5, 59))

        for i in xrange(5):
            for k in xrange(59):
                cost_arr[i, k] = np.min(results_list[i * 59 + k]['cost'][1, :])

        fn = ds.create_exp_id(network_type, "all", "all", batch_size,
                              num_epochs, training_method, regularization)
        fn = os.path.join(results_path, "images", fn+"_neurons-cost")
        draw.draw_neurons_layers_cost(cost_arr, fn)

    elif mode == "cost-regular":
        assert num_layers != "all", "num-layers should be 1-5 \
                                     in cost-regular mode"
        assert num_neurons != "all", "num-neurons should be 10-300 \
                                      in cost-regular mode"
        assert training_method != "all", "training-method shouldn't be \
                                          all in cost-regular mode"
        assert regularization == "all", "regularization should be all \
                                         in cost-regular mode"
        results_list = search(results_path, network_type, num_layers,
                              num_neurons, batch_size, num_epochs,
                              training_method, regularization)
        cost_arr = results_list[0]['cost'][1, :]
        cost_arr = np.vstack((cost_arr, results_list[1]['cost'][1, :]))
        fn = ds.create_exp_id(network_type, num_layers, num_neurons,
                              batch_size, num_epochs, training_method, "all")
        fn = os.path.join(results_path, "images", fn+"_cost-regular")
        draw.draw_cost_dropout(cost_arr, fn)
    else:
        print "Error"
    return

parser = argparse.ArgumentParser(
    description="Result Analysis for Rolling Force Prediction Problem")

parser.add_argument("--results-path", type=str,
                    default="/Users/dgyHome/Downloads/results",
                    help="Destination of result files.")
parser.add_argument("--network-type", type=str, default="feedforward",
                    help="Type of network: feedforward or RNN.")
parser.add_argument("--num-layers", type=str, default="1",
                    help="Number of hidden feedforward layers: 1-5, all")
parser.add_argument("--num-neurons", type=str, default="10",
                    help="Number of neurons: 10-300, all ")
parser.add_argument("--batch-size", type=str, default="200",
                    help="Batch size of each mini-batch samples.")
parser.add_argument("--num-epochs", type=str, default="1000",
                    help="Total training epochs for training.")
parser.add_argument("--training-method", type=str, default="sgd",
                    help="Training method: sgd, momentum, adagrad, rmsprop.")
parser.add_argument("--regularization", type=str, default="l2",
                    help="Regularization method: l2, dropout, none")
parser.add_argument("--mode", type=str, default="targets-predicted",
                    help="output mode: targets-predicted, epochs-cost, \
                         cost-algorithm, neurons-cost, cost-regular")

args = parser.parse_args()

analysis(**vars(args))
