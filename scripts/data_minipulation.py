"""
Discover usage of HDF5 datasets

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import cPickle as pickle;

import rolling.draw as draw;

f=open("../results/feedforward_1_10_200_200_sgd_l2.pkl", 'r');
data=pickle.load(f);

train_targets=data['train_targets'];
train_predicted=data['train_predicted'];
cost=data['cost'];
print train_targets.shape
print train_predicted.shape;

draw.draw_epochs_cost(cost, "test_cost_for_complete")