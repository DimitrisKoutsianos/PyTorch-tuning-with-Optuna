import engine
from model import suggest_architecture, MLP, mlp_rebuild
from data_setup import fashion_mnist_setup

import torch
from torch import nn
import torch.nn.functional as F

import torchmetrics
from torchmetrics import Accuracy, F1Score


import time
import optuna
import numpy as np
from colorama import Fore, Back, Style
from pathlib import Path

from torchinfo import summary

train_loader, val_loader, test_loader = fashion_mnist_setup()

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss().to(device)
acc_fn = Accuracy("multiclass", num_classes = 10).to(device)

def objective(trial):
  """
    Optuna objective function for hyperparameter tuning.

    This function defines the optimization objective for the hyperparameter tuning process.
    It suggests hyperparameters for the MLP model, trains the model, and returns the validation accuracy.

    Parameters:
        trial (optuna.Trial): The Optuna trial object for hyperparameter optimization.

    Returns:
        float: The validation accuracy achieved using the suggested hyperparameters.

    Example:
    ```python
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    ```
  """
  lr = trial.suggest_float("lr", 1e-5,1e-2)
  a = suggest_architecture(trial)
  model = MLP(a).to(device)

  print(summary(model=model,
        input_size=(32, 1, 28, 28), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
  ))

  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  #early_stopping = EarlyStopper(mode = "maximize")
  reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max",
                                                         verbose = True,
                                                         patience = 5,
                                                         cooldown = 2)

  val_acc = engine.train(model = model,
                  train_loader = train_loader,
                  val_loader = val_loader,
                  optimizer = optimizer,
                  reduce_lr = reduce_lr,
                  loss_fn = loss_fn,
                  epochs = 1000,
                  device = device,
                  trial = trial)

  time.sleep(0.5)

  return val_acc

study = optuna.create_study(
        direction="maximize",
    )
study.optimize(objective, n_trials=5)

print("\n\n\n\n\n\n")
print("[INFO]: Finished Hyperparameter Tuning")
print("Number of finished trials: {}".format(len(study.trials)))
trial = study.best_trial
print(f"Best trial: {trial.number}")
print("Value: {}".format(trial.value))
print(study.best_params)


print("\n\n\n\n\n\n")
print("[INFO]: Checking model performance on test data...")
model = mlp_rebuild(input_shape = 28*28,
                      classes = 10,
                      best_params = study.best_params,
                      weights_path = f"/content/models/trial_{trial.number}/Trial_{trial.number}_state_dict.pth",
                      device = device)



f1_fn = F1Score("multiclass", num_classes = 10).to(device)
test_loss, test_acc, test_f1 = engine.test_step(model = model,
                                  dataloader = test_loader,
                                  loss_fn = loss_fn,
                                  f1_fn = f1_fn,
                                  device = device)
print(f"test loss: {test_loss}, test accuracy: {test_acc*100:.2f}%, test f1: {test_f1*100:.2f}%")
