"""In charge of data preparation and analysis.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import cPickle as pickle
import numpy as np
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream


def load_rf_data(filename):
    """Load rolling force dataset.

    Parameters
    ----------
    filename : string
        path of the dataset

    Returns
    -------
    X : array
        Instances of data, size in (num_of_instances x dimension)
    Y : vector
        output variable as output force
    """
    A = np.loadtxt(filename, dtype="float32", delimiter=",")

    X = A[:, :10]
    y = A[:, -1]

    return X, y


def remove_rf_outliers(X, y):
    """Remove outliers of Rolling Force dataset.

    Parameters
    ----------
    X : array
        Inputs of data, size in (num_of_instances x dimension)
    y : array
        Outputs of data

    Returns
    -------
    X_proc : array
        Outliers removed inputs
    y_proc : array
        Outliers removed output
    """
    summary = np.percentile(y, [25, 50, 75])
    high_lim = summary[0] - 1.5 * (summary[2] - summary[1])
    low_lim = summary[2] + 1.5 * (summary[2] - summary[1])

    data = np.hstack((X, y[None].T))

    data = data[~(data[:, -1] >= low_lim)]
    data = data[~(data[:, -1] <= high_lim)]

    # remove last instances
    data = data[:-(data.shape[0] % 1000), :]

    return data[:, :-1], data[:, -1]


def transform_sequence(data_name,
                       which_sets,
                       batch_size):
    """Transform data to appropriate sequence for LSTM network.

    Parameters
    ----------
    data_name : string
        name of the dataset
    which_sets : string
        which sets is loading (train, test)
    batch_size : int
        size of each batch

    Returns
    -------
    """
    data_f = H5PYDataset(data_name, which_sets=(which_sets,),
                         sources=['features'], load_in_memory=True)

    data_t = H5PYDataset(data_name, which_sets=(which_sets,),
                         sources=['targets'], load_in_memory=True)

    data_f = data_f.data_sources
    data_f = data_f[0].reshape(1, data_f[0].shape[0], data_f[0].shape[1])
    data_t = data_t.data_sources
    data_t = data_t[0]

    num_batches = data_f.shape[1] / batch_size

    df = []
    dt = []
    for i in xrange(num_batches):
        df.append(data_f[:, i * batch_size:(i + 1) * batch_size, :])
        dt.append(data_t[i * batch_size:(i + 1) * batch_size, :])

    return {"features": df, "targets": dt}


def prepare_datastream(data_name,
                       batch_size):
    """Prepare training and testing data stream.

    Parameters
    ----------
    data_name : string
        path of rolling force dataset
    batch_size : int
        size of each mini-batch data

    Returns
    -------
    stream_train : DataStream
        training data stream
    stream_test : DataStream
        testing data stream
    """
    train_set = H5PYDataset(data_name, which_sets=(
        "train",), load_in_memory=True)
    test_set = H5PYDataset(data_name, which_sets=(
        "test", ), load_in_memory=True)

    train_scheme = SequentialScheme(train_set.num_examples,
                                    batch_size=batch_size)
    test_scheme = SequentialScheme(test_set.num_examples,
                                   batch_size=batch_size)
    stream_train = DataStream.default_stream(train_set,
                                             iteration_scheme=train_scheme)
    stream_test = DataStream.default_stream(test_set,
                                            iteration_scheme=test_scheme)

    return train_set, stream_train, test_set, stream_test


def get_data(dataset):
    """Get features and targets from a specific dataset.

    Parameters
    ----------
    dataset : Dataset object
        dataset that can provide features and targets

    Returns
    -------
    features : arrary
        features in the dataset
    targets : array
        1-d targets in the dataset
    """
    handle = dataset.open()
    data = dataset.get_data(handle, slice(0, dataset.num_examples))
    features = data[0]
    targets = data[1]
    dataset.close(handle)

    return features, targets


def get_iter_data(dataset):
    """Get data from a iterable dataset.

    Parameters
    ----------
    dataset : IterableDataset
        dataset that can provide features and targets

    Returns
    -------
    features : Dictionary
        store feature matrices
    targets : Dictionary
        store target vectors
    """
    num_samples = dataset.num_examples

    handle = dataset.open()
    features = []
    targets = []
    for i in xrange(num_samples):
        data = dataset.get_data(handle)
        features.append(data[0])
        targets.append(data[1])

    dataset.close(handle)

    targets_arr = targets[0]
    for i in xrange(1, num_samples):
        targets_arr = np.vstack((targets_arr, targets[i]))

    return features, targets_arr


def get_cost_data(monitor, num_batches, num_epochs):
    """Get cost value from a monitor.

    Parameters
    ----------
    monitor : DataStreamMonitoring
        Training monitor / testing monitor
    num_batches : int
        total number of batches
    num_epochs : int
        total number of epochs

    Returns
    cost : array
        2 rows array of cost
        1st row : train_cost
        2nd row : test_cost
    """
    log = monitor.main_loop.log

    cost = np.zeros((2, num_epochs))

    for i in xrange(num_epochs):
        cost[0, i] = log[(i + 1) * num_batches]['train_cost']
        cost[1, i] = log[(i + 1) * num_batches]['test_cost']

    return cost


def create_exp_id(exp_network,
                  num_layers,
                  num_neurons,
                  batch_size,
                  num_epochs,
                  learning_method,
                  regularization):
    """Create identifier for particular experiment.

    Parameters
    ----------
    exp_network : string
        RNN/Feedforward
    num_layers : int
        number of feedforward hidden layers
    num_neurons : int
        number of neurons for each hidden layers
        fixed for simplify situation
    batch_size : int
        size of each mini-batch
    num_epochs : int
        total number of training epochs
    learning_method : string
        SGD, momentum SGD, AdaGrad, RMSprop
    regularization : string
        Dropout / L2 regularization

    Returns
    -------
    exp_id : string
        experiment identifier
    """
    exp_id = exp_network + "_"
    exp_id = exp_id + str(num_layers) + "_"
    exp_id = exp_id + str(num_neurons) + "_"
    exp_id = exp_id + str(batch_size) + "_"
    exp_id = exp_id + str(num_epochs) + "_"
    exp_id = exp_id + learning_method + "_"
    exp_id = exp_id + regularization

    return exp_id


def save_experiment(train_targets,
                    train_predicted,
                    test_targets,
                    test_predicted,
                    cost,
                    exp_network,
                    num_layers,
                    num_neurons,
                    batch_size,
                    num_epochs,
                    learning_method,
                    regularization,
                    exp_id,
                    save_path):
    """Save experiment data.

    Parameters
    ----------
    targets : array
        1-d array of target output
    predicted : array
        1-d array of predicted output
    cost : array
        2 x num_epochs matrix
        1st row : train cost
        2nd row : test cost
    exp_network : string
        RNN/Feedforward
    num_layers : int
        number of feedforward hidden layers
    num_neurons : int
        number of neurons for each hidden layers
        fixed for simplify situation
    batch_size : int
        size of each mini-batch
    num_epochs : int
        total number of training epochs
    learning_method : string
        SGD, momentum SGD, AdaGrad, RMSprop
    regularization : string
        Dropout / L2 regularization
    exp_id : string
        identifier of experiment
    save_path : string
        save path of data, to directory level

    Returns
    -------
    A saved pkl file that records the experiment
    """
    exp_data = {}

    exp_data['train_targets'] = train_targets
    exp_data['train_predicted'] = train_predicted
    exp_data['test_targets'] = test_targets
    exp_data['test_predicted'] = test_predicted
    exp_data['cost'] = cost
    exp_data['exp_network'] = exp_network
    exp_data['num_layers'] = num_layers
    exp_data['num_neurons'] = num_neurons
    exp_data['batch_size'] = batch_size
    exp_data['num_epochs'] = num_epochs
    exp_data['learning_method'] = learning_method
    exp_data['regularization'] = regularization
    exp_data['exp_id'] = exp_id

    f = file(save_path + exp_id + ".pkl", "wb")
    pickle.dump(exp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    return
