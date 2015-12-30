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
  
  plt.figure(1);
  plt.boxplot(data);
  plt.ylabel("Rolling Force"); 
  plt.savefig("../results/"+filename+".png", dpi=75);
  plt.savefig("../results/"+filename+".eps", dpi=75);
  
  return ;