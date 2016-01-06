"""
Network configuration functions

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from blocks.algorithms import GradientDescent, Scale, Momentum, AdaGrad, RMSProp, StepClipping, CompositeRule;
from blocks.bricks import MLP, Rectifier;
from blocks.bricks.cost import SquaredError, AbsoluteError;
from blocks.filter import VariableFilter;
from blocks.graph import apply_dropout;
from blocks.graph import ComputationGraph;
from blocks.initialization import Constant, IsotropicGaussian;
from blocks.roles import INPUT, WEIGHT;
from blocks.theano_expressions import l2_norm;

def l2_regularization(cg, rate=0.01):
  """
  compute L2 regularization decay
  
  Parameters
  ----------
  cg : ComputationGraph
      computation graph for a network
  rate : float
      L2 regularization rate
      
  Returns
  -------
  L2_cost : expression
      L2 cost for a network
  """
  
  W=VariableFilter(roles=[WEIGHT])(cg.variables);
  L2_cost=rate*l2_norm(W);
  
  return L2_cost;

def dropout(cg):
  """
  create dropout computation graph
  
  Parameters
  ----------
  cg : ComputationGraph
      origin computation graph
      
  Returns
  -------
  dropout_cg : ComputationGraph
      dropped out computation graph
  """
  
  inputs=VariableFilter(roles=[INPUT])(cg.variables);
  dropout_cg=apply_dropout(cg, inputs, 0.5);
  
  return dropout_cg;

def create_cg_and_cost(y, y_hat,
                       regularization):
  """
  Create computation graph and cost
  
  Parameters
  ----------
  y : Tensor
      target output variable
  y_hat : Tensor
      predicted output variable
  regularization : string
      regularization method: dropout/l2
      
  Returns
  -------
  cost : expression
      final cost expression
  regularized_cg : ComputationGraph
      regularized computation Graph
  """
  
  cost=AbsoluteError().apply(y, y_hat);
  cg=ComputationGraph(cost);
  
  if regularization == "l2":
    cost+=l2_regularization(cg);
    regularized_cg=cg;
  elif regularization == "dropout":
    regularized_cg=dropout(cg);
  elif regularization == "none":
    regularized_cg=cg;
    
  cost.name="cost";
  
  return cost, regularized_cg;

def setup_ff_network(in_dim,
                     out_dim,
                     num_layers,
                     num_neurons):
  """
  Setup a feedforward neural network
  
  Parameters
  ----------
  in_dim : int
      input dimension of network
  out_dim : int
      output dimension of network
  num_layers : int
      number of hidden layers
  num_neurons : int
      number of neurons of each layer

  Returns
  -------
  net : object
      network structure
  """
  
  activations=[Rectifier()];
  dims=[in_dim];
  
  for i in xrange(num_layers):
    activations.append(Rectifier());
    dims.append(num_neurons);
  
  dims.append(out_dim);
  
  net=MLP(activations=activations,
          dims=dims,
          weights_init=IsotropicGaussian(),
          biases_init=Constant(0.01));
          
  return net;

def setup_algorithms(cost, cg, method, type="ff"):
  """
  setup training algorithm
  
  Parameters
  ----------
  cost : expression
      cost expression
  cg : ComputationGraph
      Computation graph
  method : string
      training method: SGD, momentum SGD, AdaGrad, RMSprop
  learning_rate : float
      learning rate for learning method
      
  Returns
  -------
  algorithm : GradientDescent
      Gradient Descent algorithm based on different optimization method
  """
  
  if method == "sgd":
    step_rule=Scale(learning_rate=0.01);
  elif method == "momentum":
    step_rule=Momentum(learning_rate=0.01, momentum=0.95);
  elif method == "adagrad":
    step_rule=AdaGrad();
  elif method == "rmsprop":
    step_rule=RMSProp();
    
  if type=="RNN":
    step_rule=CompositeRule([StepClipping(1.0), step_rule]);
  
  algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                              step_rule=step_rule);
  
  return algorithm;
