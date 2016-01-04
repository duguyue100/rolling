# Rolling

Neural Network Solution to Rolling Force Prediction

## Updates

+ Data preparation and pre-processing [2015-12-30]
+ Basic Feedforward and Recurrent networks are set up [2015-12-31]
+ Drawing figure [2016-01-02]
+ Function of constructing experiment identifier. [2016-01-03]
+ Experiment documentation code [2016-01-03]
+ Complete feedfoward network experiment setup [2016-01-04]
+ Complete RNN network experiment setup [TODO]
+ Update drawing functions [TODO]
+ Update README [TODO]

## Notes

### Data Preprocessing

The dataset consists of over 20,000 samples, therefore, the first move I've conducted is to remove outliers.
I plotted box plot and calculated summary of the data. Finally, total 18,000 samples are saved.

For this cleaned data, I removed mean and standard deviation for each sample in order to standarlized the data.
Then PCA is carried out. I found that for rotated data, first 4 dimensions are able to preserve over 99% distribution.
In this case, I removed the rest 6 dimension.

Therefore, now the input data is a matrix of `4*18000`, and output data is a vector of `1*18000`.

By the way, output data is not touched during the pre-processing. 

### Data Preparation

This project is based on [Blocks](https://github.com/mila-udem/blocks) and [Fuel](https://github.com/mila-udem/fuel).
Therefore, I wrapped processed data into a HDF5 dataset with Fuel's help. The detailed code is in `data_analysis.py`.

The dataset is split as 70% training and 30% testing. Since the originally data is not sorted,
here then I didn't randomlise the dataset. Furthermore, there is no validation set.

Here input data is called "features" and output data is called "targets".

### Experiment setup


## Contacts

Yuhuang Hu  
Email: duguyue100@gmail.com  
_No.42, North, Flatland_