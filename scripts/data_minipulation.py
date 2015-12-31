"""
Discover usage of HDF5 datasets

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from fuel.datasets.hdf5 import H5PYDataset;

data_name="../data/rf_data_PCA.hdf5";

train_set=H5PYDataset(data_name, which_sets=("train",), sources=['targets'], load_in_memory=True);
test_set=H5PYDataset(data_name, which_sets=("test", ), load_in_memory=True);

train_data = train_set.data_sources;
train_data = train_data[0].reshape(1, train_data[0].shape[0], train_data[0].shape[1]);

print train_data.shape

test_data = train_set.data_sources;
test_data = test_data[0].reshape(1, test_data[0].shape[0], test_data[0].shape[1]);