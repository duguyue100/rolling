"""
Data preparation and analysis for Rolling Force dataset

+ removed outliers (origins 20666, remains 18393) and last 393 instances
  finally 18000 instances

+ split dataset: 70% training (12600), 30% testing (5400)

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np;
import numpy.linalg as LA;
import h5py;
import matplotlib;
matplotlib.use('tkagg');
import matplotlib.pyplot as plt;

from fuel.datasets.hdf5 import H5PYDataset

import rolling.dataset as db;

# Loading dataset
X, y=db.load_rf_data("../data/rf_data.csv");
# Remove outliers
X, y=db.remove_rf_outliers(X, y);
y=y[None].T;

# Preprocess dataset
X=X-np.mean(X, axis=0);
X=X/np.std(X,axis=1).reshape((X.shape[0],1));

# PCA
X_cov=X.T.dot(X)/X.shape[0];
U, S, _ = LA.svd(X_cov);
X=U.T.dot(X.T).T;
X=X[:, :4];

# Prepare dataset

## split dataset
num_samples=X.shape[0];
n_train=num_samples*0.7;
n_test=num_samples*0.3;

## prepare hdf5 datafile 
f=h5py.File("../data/rf_data.hdf5", mode="w");

features=f.create_dataset("features",
                          shape=X.shape,
                          dtype="float32");

targets=f.create_dataset("targets",
                         shape=y.shape,
                         dtype="float32");

features[...]=X;
targets[...]=y;

features.dims[0].label="batch";
features.dims[1].label="feature";
targets.dims[0].label='batch';
targets.dims[1].label='index';

split_dict={"train" : {"features": (0, n_train), "targets": (0, n_train)},
            "test"  : {"features" : (n_train, num_samples), "targets": (n_train, num_samples)}};
  
f.attrs["split"]=H5PYDataset.create_split_array(split_dict);

f.flush();
f.close();

train_set = H5PYDataset('../data/rf_data.hdf5', which_sets=('train',));
print train_set.num_examples;

test_set = H5PYDataset('../data/rf_data.hdf5', which_sets=('test',))
print test_set.num_examples;