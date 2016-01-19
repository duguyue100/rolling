# Rolling

Neural Network Solution to Rolling Force Prediction

## Updates

+ Data preparation and pre-processing [2015-12-30]
+ Basic Feedforward and Recurrent networks are set up [2015-12-31]
+ Drawing figure [2016-01-02]
+ Function of constructing experiment identifier. [2016-01-03]
+ Experiment documentation code [2016-01-03]
+ Complete feedfoward network experiment setup [2016-01-04]
+ Complete RNN network experiment setup [2016-01-05]
+ Update drawing functions [2015-01-05]
+ Update README [2016-01-05]
+ Update RNN by clipping the weights [2016-01-07] [refer to [this](https://github.com/mila-udem/blocks-examples/blob/master/reverse_words/__init__.py#L199-L201)]
+ Write some searching functions [2016-01-05]
+ Update find best functions [2016-01-19]

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

Each experiment is determined by several factors:

+ the type of network: Feedforward Network or RNN Network
+ input dimension, in here it's fixed: 4
+ output dimension, in here it's fixed too: 1
+ number of feedforward hidden layers, since we just use single LSTM layer in RNN network, so it's best to feedforward layers to quantify depth.
+ number of neurons: to simplify the problem, each feedforward layer has same number of neurons.
+ training method: all experiments are trained with Gradient-based optimization algorithms, here we compare 4 of them: SGD, momentum SGD, AdaGrad and RMSprop.
+ regularization method: we compare L2 regularization and dropout in experiments.
+ batch size: number of samples in each mini-batch.
+ number of epochs: total training steps of a experiment.

Based on above information, each experiment is identified by one unique ID, the format is as following:

```
[Feedforward/RNN]_[number of hidden layers]_[number of neurons]_[batch size]_[number of epochs]_[training algorithm]_[regularization] 
```

Each experiment is documented by a `pkl` file by using `cPickle` package. Each file saves all relevant output information:

|Documented Data                              |Identifier       |
|---------------------------------------------|-----------------|
|target output of training set                |`train_targets`  |
|predicted output of training set             |`train_predicted`|
|target output of testing set                 |`test_targets`   |
|predicted output of testing set              |`test_predicted` |
|cost matrix for both training and testing set|`cost`           |
|type of network                              |`exp_network`    |
|number of feedforward hidden layers          |`num_layers`     |
|number of neurons                            |`num_neurons`    |
|batch size for each mini-batch               |`batch_size`     |
|number of training epochs                    |`num_epochs`     |
|learning algorithm method                    |`learning_method`|
|regularization method                        |`regularization` |
|experiment Identifier                        |`exp_id`         |

### Conduct Experiment

Open a terminal and key in following commands (assume you have the same structure as mine):

```
cd workspace/rolling/scripts
```

For Feedforward Regression experiment:

```
python feedforward_exp.py with ../configs/ff_regression_rf_max.json
```

For LSTM Regression experiment:

```
python lstm_exp.py with ../configs/lstm_regression_rf_max.json
```

### Result analysis

I wrote a script that automatically analyzes the results and produce the graph.
The script takes several arguments, and will produce 5 kinds of graph for presenting the results.

Further will include summary of the results if needed.

```
$ python result-analysis.py -h
usage: result-analysis.py [-h] [--results-path RESULTS_PATH]
                          [--network-type NETWORK_TYPE]
                          [--num-layers NUM_LAYERS]
                          [--num-neurons NUM_NEURONS]
                          [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS]
                          [--training-method TRAINING_METHOD]
                          [--regularization REGULARIZATION] [--mode MODE]

Result Analysis for Rolling Force Prediction Problem

optional arguments:
  -h, --help            show this help message and exit
  --results-path RESULTS_PATH
                        Destination of result files.
  --network-type NETWORK_TYPE
                        Type of network: feedforward or RNN.
  --num-layers NUM_LAYERS
                        Number of hidden feedforward layers: 1-5, all
  --num-neurons NUM_NEURONS
                        Number of neurons: 10-300, all
  --batch-size BATCH_SIZE
                        Batch size of each mini-batch samples.
  --num-epochs NUM_EPOCHS
                        Total training epochs for training.
  --training-method TRAINING_METHOD
                        Training method: sgd, momentum, adagrad, rmsprop.
  --regularization REGULARIZATION
                        Regularization method: l2, dropout, none
  --mode MODE           output mode: targets-predicted, epochs-cost, cost-
                        algorithm, neurons-cost, cost-regular
```

## Contacts

Yuhuang Hu  
Email: duguyue100@gmail.com  
_No.42, North, Flatland_