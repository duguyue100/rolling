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

from blocks.bricks import MLP;
from blocks.bricks import Linear, Rectifier, Tanh;
from blocks.bricks.cost import SquaredError, AbsoluteError;
from blocks.initialization import Constant, IsotropicGaussian;
from blocks.graph import ComputationGraph;
from blocks.algorithms import GradientDescent, Adam, AdaGrad;
from blocks.extensions import FinishAfter, Printing, ProgressBar;
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint;
from blocks.main_loop import MainLoop;

from fuel.streams import DataStream;
from fuel.schemes import SequentialScheme;
from fuel.datasets.hdf5 import H5PYDataset;

import rolling.dataset as ds;
import rolling.draw as draw;

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
  train_set=H5PYDataset(data_name, which_sets=("train",), load_in_memory=True);
  test_set=H5PYDataset(data_name, which_sets=("test", ), load_in_memory=True);
   
  stream_train=DataStream.default_stream(train_set,
                  iteration_scheme=SequentialScheme(train_set.num_examples, batch_size=batch_size));
  stream_test=DataStream.default_stream(test_set,
                  iteration_scheme=SequentialScheme(test_set.num_examples, batch_size=batch_size));            
  
  X=T.matrix("features");
  y=T.matrix("targets");
  
  net=MLP(activations=[Tanh(), Rectifier(), Rectifier(), Rectifier(),],
          dims=[in_dim, hid_dim, hid_dim, hid_dim, out_dim],
          weights_init=IsotropicGaussian(),
          biases_init=Constant(0.01));
  y_hat=net.apply(X);
  
  cost=AbsoluteError().apply(y, y_hat);
  cost.name="cost";
  
  net.initialize();
  
  cg = ComputationGraph(cost);
  
  algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                              step_rule=AdaGrad());
                              
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
  
  
  
  
  