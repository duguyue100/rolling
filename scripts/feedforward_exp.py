"""
Setup experiment: Feedforward Network Regression

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import sys;
from blocks.graph import apply_dropout
sys.path.append("..");

from sacred import Experiment;

import theano;
import theano.tensor as T;

from blocks.extensions import FinishAfter, Printing, ProgressBar;
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint;
from blocks.main_loop import MainLoop;

import rolling.dataset as ds;
import rolling.draw as draw;
import rolling.netconfigs as nc;

exp=Experiment("Rolling Force - FeedForward Regression")

@exp.config
def rf_ff_config():
  data_name="";
  in_dim=0;
  hid_dim=0;
  out_dim=0;
  batch_size=0;
  num_epochs=0;
  
@exp.automain
def rf_ff_experiment(data_name,
                     in_dim,
                     hid_dim,
                     out_dim,
                     batch_size,
                     num_epochs):
  # load dataset
  train_set, stream_train, test_set, stream_test=ds.prepare_datastream(data_name, batch_size);
    
  # setup network
  X=T.matrix("features");
  y=T.matrix("targets");  
  net=nc.setup_ff_network(in_dim, out_dim, 4, hid_dim);  
  y_hat=net.apply(X);
  cost, cg=nc.create_cg_and_cost(y, y_hat, "dropout");
  net.initialize();
  
  algorithm=nc.setup_algorithms(cost, cg, "adagrad", 0.002);                              
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
  
  ## construct experiment identifier
  ## 1. RNN/Feedfoward
  ## 2. number of hidden layers (optional)
  ## 3. number of neurons
  ## 4. learning method (optional)
  ## 5. learning rate
  ## 6. dropout/L2 regularization

  ## TODO wirte a function that produce proper id
  
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
  
  # Drawing
  
  draw.draw_epochs_cost(cost, "testing");
  draw.draw_target_predicted(train_targets, train_predicted, "train_test");
  draw.draw_target_predicted(test_targets, test_predicted, "test_test");