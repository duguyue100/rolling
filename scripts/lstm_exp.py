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

import theano.tensor as T;

from blocks.bricks import MLP;
from blocks.bricks import Linear, Rectifier;
from blocks.bricks.recurrent import LSTM;
from blocks.bricks.cost import SquaredError, AbsoluteError;
from blocks.initialization import Constant, IsotropicGaussian;
from blocks.graph import ComputationGraph;
from blocks.algorithms import GradientDescent, Adam, AdaGrad;
from blocks.extensions import FinishAfter, Printing, ProgressBar;
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop;

from fuel.streams import DataStream;
from fuel.datasets import IterableDataset;
from fuel.schemes import SequentialScheme, ShuffledScheme;
from fuel.transformers import Mapping;
from fuel.datasets.hdf5 import H5PYDataset;

import rolling.dataset as ds;

exp=Experiment("Rolling Force - LSTM Regression")

@exp.config
def rf_lstm_config():
  data_name="";
  in_dim=0;
  lstm_dim=0;
  out_dim=0;
  batch_size=0;
  num_epochs=0;
  
@exp.automain
def rf_lstm_experiment(data_name,
                       in_dim,
                       lstm_dim,
                       out_dim,
                       batch_size,
                       num_epochs):
  # load dataset  
  train_set=IterableDataset(ds.transform_sequence(data_name, "train", batch_size));
  test_set=IterableDataset(ds.transform_sequence(data_name, "test", batch_size));
   
  stream_train=DataStream(dataset=train_set);
  stream_test=DataStream(dataset=test_set);            
  
  X=T.tensor3("features");
  y=T.matrix("targets");
  
  x_to_h = Linear(in_dim, lstm_dim * 4, name='x_to_h',
                  weights_init=IsotropicGaussian(),
                  biases_init=Constant(0.0));
  lstm = LSTM(lstm_dim, name='lstm',
              weights_init=IsotropicGaussian(),
              biases_init=Constant(0.0));
  h_to_o=MLP(activations=[Rectifier(), Rectifier(), Rectifier()],
             dims=[lstm_dim, lstm_dim, lstm_dim, out_dim],
             weights_init=IsotropicGaussian(),
             biases_init=Constant(0.01));

  X_trans=x_to_h.apply(X);
  h, c=lstm.apply(X_trans);
  y_hat=h_to_o.apply(h[-1]);
  
  cost=AbsoluteError().apply(y, y_hat);
  cost.name="cost";
  
  lstm.initialize();
  x_to_h.initialize();
  h_to_o.initialize();
  
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
  
  
  
  
  