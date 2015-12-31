"""
In charge of data preparation and analysis.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np;
from fuel.datasets.hdf5 import H5PYDataset;

def load_rf_data(filename):
  """
  Load rolling force dataset
  
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
  
  A=np.loadtxt(filename, dtype="float32", delimiter=",");
  
  X=A[:,:10];
  y=A[:,-1];
  
  return X, y;

def remove_rf_outliers(X, y):
  """
  Remove outliers of Rolling Force dataset
  
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
  
  summary=np.percentile(y, [25, 50, 75]);
  high_lim=summary[0]-1.5*(summary[2]-summary[1]);
  low_lim=summary[2]+1.5*(summary[2]-summary[1]);
  
  data=np.hstack((X, y[None].T));
  
  data = data[~(data[:,-1]>=low_lim)];
  data = data[~(data[:,-1]<=high_lim)];
  
  # remove last instances
  data=data[:-(data.shape[0]%1000), :]
  
  return data[:, :-1], data[:,-1];

def transform_sequence(data_name,
                       which_sets,
                       batch_size):
  """
  Transform data to appropriate sequence for LSTM network
  
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
  
  data_f=H5PYDataset(data_name, which_sets=(which_sets,), 
                     sources=['features'], load_in_memory=True);
    
  data_t=H5PYDataset(data_name, which_sets=(which_sets,), 
                     sources=['targets'], load_in_memory=True);

  data_f=data_f.data_sources;
  data_f=data_f[0].reshape(1, data_f[0].shape[0], data_f[0].shape[1]);
  data_t=data_t.data_sources;
  data_t=data_t[0];
  
  num_batches=data_f.shape[1]/batch_size;

  df=[]; dt=[];
  for i in xrange(num_batches):
    df.append(data_f[:, i*batch_size:(i+1)*batch_size, :]);
    dt.append(data_t[i*batch_size:(i+1)*batch_size, :]);
  
  return {"features": df, "targets": dt};







