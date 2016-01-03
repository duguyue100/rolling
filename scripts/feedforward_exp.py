"""
Setup experiment: Feedforward Network Regression

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import sys;
sys.path.append("..");

from sacred import Experiment;

import theano;
import theano.tensor as T;

from blocks.extensions import FinishAfter, Printing, ProgressBar;
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop;

import rolling.dataset as ds;
import rolling.netconfigs as nc;

exp=Experiment("Rolling Force - FeedForward Regression")

@exp.config
def rf_ff_config():
  data_name="";      # dataset name in string 
  exp_network="";    # type of neural netowrks
  in_dim=0;          # input dimension of network
  out_dim=0;         # output dimension of network
  num_layers=0;      # maximum number of feedforward hidden layers
  num_neurons=0;     # maximum limit of number of neurons
  batch_size=0;      # batch size of each mini-batch
  num_epochs=0;      # total number of epochs
  
@exp.automain
def rf_ff_experiment(data_name,
                     exp_network,
                     in_dim,
                     out_dim,
                     num_layers,
                     num_neurons,
                     batch_size,
                     num_epochs):
  # load dataset
  train_set, stream_train, test_set, stream_test=ds.prepare_datastream(data_name, batch_size);
  methods=['sgd', 'momentum', 'adagrad', 'rmsprop'];
  regularization=['l2', 'dropout']; # regularization in list
  
  for n_layers in xrange(1, num_layers+1):
    for n_neurons in xrange(10, num_neurons+5, 5):
      for method in methods:
        for regular in regularization:
          # setup network
          X=T.matrix("features");
          y=T.matrix("targets");  
          net=nc.setup_ff_network(in_dim, out_dim, n_layers, n_neurons);  
          y_hat=net.apply(X);
          cost, cg=nc.create_cg_and_cost(y, y_hat, regular);
          net.initialize();
  
          algorithm=nc.setup_algorithms(cost, cg, method);                              
          test_monitor = DataStreamMonitoring(variables=[cost],
                                              data_stream=stream_test, prefix="test")
          train_monitor = TrainingDataMonitoring(variables=[cost], prefix="train",
                                                 after_epoch=True)
  
          main_loop = MainLoop(algorithm=algorithm,
                               data_stream=stream_train,
                               extensions=[test_monitor, train_monitor,
                                           FinishAfter(after_n_epochs=num_epochs),
                                           Printing(), ProgressBar()])
  
          main_loop.run()
  
          # Saving results
          exp_id=ds.create_exp_id(exp_network, n_layers, n_neurons, batch_size, 
                                  num_epochs, method, regular);
    
          ## prepare related functions
          predict=theano.function([X], y_hat);
  
          ## prepare related data
          train_features, train_targets=ds.get_data(train_set);
          test_features, test_targets=ds.get_data(test_set);
  
          ## Prediction of result
          train_predicted=predict(train_features);
          test_predicted=predict(test_features);
  
          ## Get cost
          cost=ds.get_cost_data(test_monitor, train_set.num_examples/batch_size, num_epochs);
  
          # logging
  
          ds.save_experiment(train_targets, train_predicted, test_targets, test_predicted, 
                             cost, exp_network, n_layers, n_neurons,
                             batch_size, num_epochs, method, regular,
                             exp_id, "../results/");  