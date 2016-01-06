"""
Setup experiment: using LSTM for regression

should get this to work based on 3d tensor instead of 2d since it's dealing with sequences
however, what I'm not sure is to make it as a sequence of 1 or sequence of total size of data
I guess the first one is more reasonable

Author: Yuhuang Hu
Email : duguyue100@gmail.com 
"""

import sys;
sys.path.append("..");

from sacred import Experiment;

import numpy as np;
import theano;
import theano.tensor as T;

from blocks.bricks import Linear;
from blocks.bricks.recurrent import LSTM;
from blocks.initialization import Constant, IsotropicGaussian;
from blocks.extensions import FinishAfter, Printing, ProgressBar;
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop;

from fuel.streams import DataStream;
from fuel.datasets import IterableDataset;

import rolling.dataset as ds;
import rolling.netconfigs as nc;

def gen_prediction(predict,
                   features):
  """
  Generate prediction from set of features
  
  Parameters
  ----------
  predict : function
      a function of target prediction
  features : Dictionary
      a dictionary of features
      
  Returns
  -------
  predicted_arr : array
      vector of prediction
  """
  
  predicted=[];
  for x in features:
    predicted.append(predict(x));
  
  predicted_arr=predicted[0];
  for i in xrange(1, len(features)):
    predicted_arr=np.vstack((predicted_arr, predicted[0]));
    
  return predicted_arr;

exp=Experiment("Rolling Force - LSTM Regression")

@exp.config
def rf_lstm_config():
  data_name="";      # dataset name in string 
  exp_network="";    # type of neural netowrks
  in_dim=0;          # input dimension of network
  out_dim=0;         # output dimension of network
  num_layers=0;      # maximum number of feedforward hidden layers
  num_neurons=0;     # maximum limit of number of neurons
  start_neurons=0;   # start point of setting number of neurons
  batch_size=0;      # batch size of each mini-batch
  num_epochs=0;      # total number of epochs
  
@exp.automain
def rf_lstm_experiment(data_name,
                       exp_network,
                       in_dim,
                       out_dim,
                       num_layers,
                       start_neurons,
                       num_neurons,
                       batch_size,
                       num_epochs):
  # load dataset  
  train_set=IterableDataset(ds.transform_sequence(data_name, "train", batch_size));
  test_set=IterableDataset(ds.transform_sequence(data_name, "test", batch_size));
  stream_train=DataStream(dataset=train_set);
  stream_test=DataStream(dataset=test_set);
  methods=['sgd', 'momentum', 'adagrad', 'rmsprop'];
  
  for n_layers in xrange(1, num_layers+1):
    for n_neurons in xrange(start_neurons, num_neurons+5, 5):
      for method in methods:            
        X=T.tensor3("features");
        y=T.matrix("targets");
  
        x_to_h = Linear(in_dim, n_neurons * 4, name='x_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0));
        lstm = LSTM(n_neurons, name='lstm',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0));
              
        h_to_o=nc.setup_ff_network(n_neurons, out_dim, n_layers-1, n_neurons);

        X_trans=x_to_h.apply(X);
        h, c=lstm.apply(X_trans);
        y_hat=h_to_o.apply(h[-1]);
        cost, cg=nc.create_cg_and_cost(y, y_hat, "none");
    
        lstm.initialize();
        x_to_h.initialize();
        h_to_o.initialize();
    
        algorithm=nc.setup_algorithms(cost, cg, method, type="RNN");
   
        test_monitor = DataStreamMonitoring(variables=[cost],
                                            data_stream=stream_test, prefix="test")
        train_monitor = TrainingDataMonitoring(variables=[cost], prefix="train",
                                               after_epoch=True)
  
        main_loop = MainLoop(algorithm=algorithm,
                             data_stream=stream_train,
                             extensions=[test_monitor, train_monitor,
                                         FinishAfter(after_n_epochs=num_epochs),
                                         Printing(), ProgressBar()])
  
        main_loop.run();
  
        # Saving results
        exp_id=ds.create_exp_id(exp_network, n_layers, n_neurons, batch_size, 
                                num_epochs, method, "none");
    
        ## prepare related functions
        predict=theano.function([X], y_hat);
  
        ## prepare related data
        train_features, train_targets=ds.get_iter_data(train_set);
        test_features, test_targets=ds.get_iter_data(test_set);
  
        ## Prediction of result
        train_predicted=gen_prediction(predict, train_features);
        test_predicted=gen_prediction(predict, test_features);
  
        ## Get cost
        cost=ds.get_cost_data(test_monitor, train_set.num_examples, num_epochs);
  
        # logging
        ds.save_experiment(train_targets, train_predicted, test_targets, test_predicted, 
                           cost, exp_network, n_layers, n_neurons,
                           batch_size, num_epochs, method, "none",
                           exp_id, "../results/");  