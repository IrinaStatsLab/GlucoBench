# GluNet

Table of Contents:

- [1. Motivation](#1-motivation)
- [2. Overview](#2-overview)

## 1. Motivation

*The purpose* of the repository is twofold: 1) collect and pre-process glucose data sets 2) establish baseline performance. *The goal* is to create a unified self-contained repository which would be a starting place for researchers and practioners to compare and develop old and new methods. *An example* of successful implementation of a similar repository for tabular data sets can be found [here](https://github.com/Yura52/tabular-dl-revisiting-models).

## 2. Overview

The main goal of the project is to create a unified library of glucose data sets and establish simple benchmarks for prediction. The current tasks are:

- [x] Familiarize with the problem and the setup. For this you can use the papers included in [here](./papers).
- [x] Get to know how we want to structure the data. Each data set is going to come in a different format. We want to unify their format and prepare them for easy use with the major ML libraries such as PyTorch. For an example of how ready-to-use PyTorch datasets look like, see [here](./example_dataset/). 
- [x] Split data sets between team member and investigate each data set.
- [x] Create data formatters for each data set, which are going to load the data set.
- [x] Create interpolate function to interpolate "small" gaps in glucose readings.
- [x] Create a splitting function splits the data set into the train, validation, and test sets.
- [ ] Implement a simple feed-forward (MLP) network for each data set that trains on the observed data and predicts multiple data points into the future. 
- [ ] The simple MLP does not make use of the known-into-the-future data (e.g. date), hence, we need more expressive architectures such as the RNN and the Transformer.
    - [ ] For seq2seq RNN-type of architectures, see [1](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks), [2](https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM), [3](https://github.com/lkulowski/LSTM_encoder_decoder). In particular [3] has a ready-to-go implementation in PyTorch.
    - [ ] For the Transformer, see [1](https://jalammar.github.io/illustrated-transformer/). 
- [ ] Now that we have implemnted and tested out the models, the key questions is whether they are performing at their best or not. In reality we nevere know how much we can squeee out of a model, hyper-parameter search is long and tedious and never guarantees anything. Fortunately, we can delegate / automate this with the use of automatic tunrs such as [Optuna](https://optuna.org/). Given the ranges for the hyperparameters, Optuna is going to try and search for the best ones.


