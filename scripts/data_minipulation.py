"""
Discover usage of HDF5 datasets

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from blocks.extensions.saveload import Load;

data=Load("results");

print data.load_iteration_state