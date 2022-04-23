# ALD saturation datasets

This repository contains simple datasets to predict optimal saturation times for an ALD process from a single growth profile and dose time.

## Dataset description

Each dataset comprises the following files:

- `profiles_train_N.npy`: file containing the dose times for the training dataset
- `dose_train_N.npy`: file containing the dose times for the training dataset
- `sat_train_N.npy`: file containing the target saturation times for the
   training dataset
- `profiles_test_N.npy`: file containing the dose times for the testing dataset
- `dose_test_N.npy`: file containing the dose times for the testing dataset
- `sat_test_N.npy`: file containing the target saturation times for the
   testing dataset

Here `N` represents the number of thickness values captured in each growth
profile.

## Utility and example files

In addition to the data, this repository contains a working example of training
various networks to the dataset:

- `readdataset.py` is a Python module containing custom Pytorch Dataset classes
  that load each of the datasets.
- `aldnets.py` contains examples of neural networks used in the training examples
- `train_shallow.py` provides a training example using Pytorch to train a shallow
  neural network in each of the datasets.
- `train_1deep.py` provides a training example using Pytorch to train networks
  with one hidden layer of various sizes in each of the datasets.
- `train_2deep.py` provides a training example using Pytorch to train networks
  with two hidden layes in each of the datasets.

