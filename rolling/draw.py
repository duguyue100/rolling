"""
Some drawing functions for producing right figures

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np;
import matplotlib;
matplotlib.use('tkagg');
import matplotlib.pyplot as plt;

def draw_rf_boxplot(data, filename):
  """
  Draw box plot for output rolling force
  
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
  
  plt.figure();
  plt.boxplot(data);
  plt.ylabel("Rolling Force"); 
  plt.savefig("../results/"+filename+".png", dpi=75);
  plt.savefig("../results/"+filename+".eps", dpi=75);
  
  return ;

def draw_epochs_cost(cost, filename):
  """
  Draw figure of epochs vs cost for both training and testing set
  
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
  
  num_epochs=cost.shape[1];
  x=np.array(xrange(num_epochs))+1;

  plt.figure();
  plt.plot(x,cost[0,:], '-', x, cost[1,:], '-.');
  plt.ylabel("cost");
  plt.xlabel("epochs");
  plt.savefig("../results/"+filename+".png", dpi=75);
  plt.savefig("../results/"+filename+".eps", dpi=75);
  
  return ;

def draw_target_predicted(target, predicted, filename):
  """
  Draw figure of target output vs predicted output
  
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
  
  num_samples=target.shape[0];
  data=np.zeros((1,2));
  for i in xrange(num_samples):
    if np.abs(target[i,0]-predicted[i,0])<100:
      tmp=np.array([target[i,0], predicted[i,0]]);
      data=np.vstack((data, tmp));
      
  data=data[1:, :];
  plt.figure();  
  plt.plot(data[:,0], data[:,1], '.');
  plt.xlabel("Target outputs");
  plt.ylabel("Predicted outputs");
  plt.savefig("../results/"+filename+".png", dpi=75);
  plt.savefig("../results/"+filename+".eps", dpi=75);
  
  return ;

def draw_cost_algorithms(cost, filename):
  """
  Draw figure of cost vs training algorithm
  
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
  
  num_epochs=cost.shape[1];
  x=np.array(xrange(num_epochs))+1;
  
  plt.figure();
  plt.plot(x, cost[0,:], '-');  # SGD
  plt.plot(x, cost[1,:], '--'); # momentum SGD
  plt.plot(x, cost[2,:], '-.'); # AdaGrad
  plt.plot(x, cost[3,:], ':');  # RMSprop
  plt.ylabel("cost");
  plt.xlabel("epochs");
  plt.savefig("../results/"+filename+".png", dpi=75);
  plt.savefig("../results/"+filename+".eps", dpi=75);
  
  return ;

def draw_neurons_layers_cost(cost, filename):
  """
  Draw figures number of neurons vs cost
  All for feedforward layer.
  
  Parameters
  ----------
  cost : array
      (k layers) x (max number of neurons) matrix in order
      1st row : 1 hidden layer
      2nd row : 2 hidden layers
      3rd row : 3 hidden layers
      4rd row : 4 hidden layers
  filename : string
      filename you want to save
      
  Returns
  -------
  save figure in png and eps format
  """
  num_neurons=cost.shape[1];
  x=np.array(xrange(num_neurons))+1;
  
  plt.figure();
  plt.plot(x, cost[0,:], '-');  # 1 hidden layer
  plt.plot(x, cost[1,:], '--'); # 2 hidden layers
  plt.plot(x, cost[2,:], '-.'); # 3 hidden layers
  plt.plot(x, cost[3,:], ':');  # 4 hidden layers
  plt.ylabel("cost");
  plt.xlabel("number of neurons");
  plt.savefig("../results/"+filename+".png", dpi=75);
  plt.savefig("../results/"+filename+".eps", dpi=75);
  
  
  return ;
  
def draw_cost_dropout(cost, filename):
  """
  Draw figure of number of epochs vs cost
  difference is with or without dropout
  
  Parameters
  ----------
  cost : arrray
      2 x number of epochs matrix in order
      1st row : without dropout
      2nd row : with dropout
  filename : string
      filename you want to save
  """
  
  num_eopchs=cost.shape[1];
  x=np.array(xrange(num_eopchs))+1;
  
  plt.figure();
  plt.plot(x, cost[0,:], '-'); # without dropout
  plt.plot(x, cost[1,:], '--'); # with dropout
  plt.ylabel("cost");
  plt.xlabel("eopchs");
  plt.savefig("../results/"+filename+".png", dpi=75);
  plt.savefig("../results/"+filename+".eps", dpi=75);
  
  return ;
  
  
  
  