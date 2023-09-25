import torch
from torchmetrics import Accuracy, F1Score
import optuna
from pathlib import Path
from colorama import Fore, Back, Style
from early_stopping import EarlyStopper

def train_step(model, dataloader, loss_fn, f1_fn, optimizer, device):
  """
    Perform a single training step for a neural network model.

    This function runs one epoch of training using the provided data and returns the training loss,
    accuracy, and F1 score.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        loss_fn (torch.nn.Module): The loss function to compute the training loss.
        f1_fn (torchmetrics.F1): The F1 score metric to compute the training F1 score.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters.
        device (str): The device (e.g., 'cpu' or 'cuda') on which to perform computations.

    Returns:
        float: Training loss for the epoch.
        float: Training accuracy for the epoch.
        float: Training F1 score for the epoch.

    Example:
    ```python
    train_loss, train_acc, train_f1 = train_step(model, train_loader, loss_fn, f1_fn, optimizer, device)
    ```
  """
  model.train()
  train_loss, train_acc, train_f1 = 0, 0, 0
  for batch, (x,y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)

    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (acc==y).sum().item()/len(y_pred)

    f1 = f1_fn(y_pred, y)
    train_f1 += f1

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  train_f1 /= len(dataloader)

  return train_loss, train_acc, train_f1


def test_step(model, dataloader, loss_fn, f1_fn, device):
  """
    Perform a single evaluation step for a neural network model.

    This function runs one epoch of evaluation using the provided data and returns the validation loss,
    accuracy, and F1 score.

    Parameters:
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        loss_fn (torch.nn.Module): The loss function to compute the validation loss.
        f1_fn (torchmetrics.F1): The F1 score metric to compute the validation F1 score.
        device (str): The device (e.g., 'cpu' or 'cuda') on which to perform computations.

    Returns:
        float: Validation loss for the epoch.
        float: Validation accuracy for the epoch.
        float: Validation F1 score for the epoch.

    Example:
    ```python
    val_loss, val_acc, val_f1 = test_step(model, val_loader, loss_fn, f1_fn, device)
    ```
  """
  model.eval()
  val_loss, val_acc, val_f1 = 0, 0, 0
  with torch.inference_mode():
    for x,y in dataloader:
      x, y = x.to(device), y.to(device)

      y_pred = model(x)
      loss = loss_fn(y_pred,y)
      val_loss += loss.item()

      acc = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      val_acc += (acc==y).sum().item()/len(y_pred)

      f1 = f1_fn(y_pred, y)
      val_f1 += f1

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    val_f1 /= len(dataloader)

  return val_loss, val_acc, val_f1

def train(model, train_loader, val_loader, optimizer, reduce_lr, loss_fn, epochs, device, trial):
  """
    Train a neural network model on a training dataset and evaluate it on a validation dataset.

    This function trains a model for the specified number of epochs and saves the best model checkpoint based on
    validation accuracy. It also prints training and validation metrics for each epoch.

    Parameters:
        model (torch.nn.Module): The neural network model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters.
        reduce_lr (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler based on validation performance.
        loss_fn (torch.nn.Module): The loss function to compute training and validation losses.
        epochs (int): The number of training epochs.
        device (str): The device (e.g., 'cpu' or 'cuda') on which to perform computations.
        trial (optuna.Trial): The Optuna trial object for hyperparameter optimization.

    Returns:
        float: The best validation accuracy achieved during training.

    Example:
    ```python
    best_val_acc = train(model, train_loader, val_loader, optimizer, reduce_lr, loss_fn, epochs, device, trial)
    ```
  """

  early_stopping = EarlyStopper(mode = "maximize")
  f1_fn = F1Score("multiclass", num_classes = 10).to(device)


  for epoch in range(epochs):
    train_loss, train_acc, train_f1 = train_step(model = model,
                                       dataloader = train_loader,
                                       optimizer = optimizer,
                                       loss_fn = loss_fn,
                                       f1_fn = f1_fn,
                                       device = device)
    val_loss, val_acc, val_f1 = test_step(model = model,
                                  dataloader = val_loader,
                                  loss_fn = loss_fn,
                                  f1_fn = f1_fn,
                                  device = device)
    reduce_lr.step(val_acc)

    MODEL_PATH = Path(f"models/trial_{trial.number}")
    MODEL_PATH.mkdir(parents = True,
                     exist_ok = True)

    MODEL_NAME = f"Trial_{trial.number}_state_dict.pth"

    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    early_stopping.save_best_weights(path = MODEL_SAVE_PATH,
                                     model = model,
                                     metric = val_acc)

    if early_stopping.early_stop(val_acc):
      print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc*100:.3f}%, Train F1: {train_f1*100:.3f}% | Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_acc*100:.3f}%, Validation F1: {val_f1*100:.3f}%")
      print(Fore.BLACK, Back.YELLOW + "[INFO]: Early Stoping" + Style.RESET_ALL)
      break
    else:
      print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc*100:.3f}%, Train F1: {train_f1*100:.3f}% | Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_acc*100:.3f}%, Validation F1: {val_f1*100:.3f}%")


  model.to("cpu")
  model.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
  model = model.to(device)
  val_loss, val_acc, val_f1 = test_step(model = model,
                                  dataloader = val_loader,
                                  loss_fn = loss_fn,
                                  f1_fn = f1_fn,
                                  device = device)

  return val_acc
