# PyTorch-tuning-with-Optuna

This is a short personal project implemented in Python. The purpose of this project is to showcase how one can use the `Optuna` to perform hyperparameter tuning on a PyTorch model.

The repository contains the following files:
- `early_stopping.py` which contains my implementation of the early stopping criterion for any PyTorch model with added support for restoration of the best weights when the training stops
- `model.py` which contains a function that creates a tune-able MLP model
- `engine.py` which contains the functions necessary for training the MLP model
- `data_setup.py` which contains the function that creates the DataLoaders of the FashionMNIST dataset
- `main.py` which is the executable of the project.
