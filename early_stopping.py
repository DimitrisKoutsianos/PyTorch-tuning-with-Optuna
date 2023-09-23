import numpy as np
from colorama import Fore, Back, Style
import torch

class EarlyStopper:
  """
    Implementing the Early Stopping Criterion for a PyTorch model.

    Args:
        mode (str): The mode to determine whether to maximize or minimize the monitored metric.
                    Must be either 'maximize' or 'minimize'.
        patience (int, optional): The number of consecutive epochs with no improvement in the
                                   monitored metric before early stopping is triggered. Default is 10.

    Attributes:
        mode (str): The mode to determine whether to maximize or minimize the monitored metric.
        patience (int): The number of consecutive epochs with no improvement required for early stopping.
        counter (int): Keeps track of consecutive epochs with no metric improvement.
        min_val_metric (float): The minimum value observed for the monitored metric.
        max_val_metric (float): The maximum value observed for the monitored metric.

    Methods:
        early_stop(self, metric):
            Monitor the metric during training and decide whether to perform early stopping.

        save_best_weights(self, path, model, metric):
            Save the model's weights if the monitored metric improves and meets the early stopping criteria.
    """
  def __init__(self, mode:str, patience:int = 10):
    self.patience = patience
    self.counter = 0
    self.min_val_metric = np.inf
    self.max_val_metric = -np.inf
    assert mode == "maximize" or mode == "minimize", f"Direction should be either 'maximize' or 'minimize', not {mode}.\nTry again..."
    self.mode = mode

  def early_stop(self, metric):
    """
        Monitor a metric during training and decide whether to perform early stopping.

        This function tracks the given `metric` value and compares it to a reference value
        (`max_val_metric` for 'maximize' mode or `min_val_metric` for 'minimize' mode).
        It keeps a counter to track consecutive epochs where the `metric` does not improve
        based on the selected mode. If the counter exceeds the `patience` threshold,
        early stopping is triggered.

        Args:
            metric (float): The current value of the metric to be monitored.

        Returns:
            bool: True if early stopping should be performed, False otherwise.
        """
    if self.mode == "maximize":
      if metric > self.max_val_metric:
        self.max_val_metric = metric
        self.counter = 0
      elif metric <= self.max_val_metric:
        self.counter += 1
        if self.counter >= self.patience:
          return True
      return False

    elif self.mode == "minimize":
      if metric < self.min_val_metric:
        self.min_val_metric = metric
        self.counter = 0
      elif metric >= self.min_val_metric:
        self.counter += 1
        if self.counter >= self.patience:
          return True
      return False


  def save_best_weights(self, path, model, metric):
    """
        Save the model's weights if the monitored metric improves and meets the early stopping criteria.

        This function saves the model's weights to the specified `path` if the monitored metric improves
        based on the selected mode ('maximize' or 'minimize'). It updates the saved weights only if the
        metric value surpasses the current best value.

        Args:
            path (str): The path to save the model's weights.
            model: The PyTorch model to save the weights from.
            metric (float): The current value of the monitored metric.

        Note:
            The `mode` attribute of the object determines whether the metric should be maximized or minimized.

        Example:
            early_stopper = EarlyStopper(mode='maximize')
            for epoch in range(epochs):
                train_loss, val_loss = train_and_evaluate()
                if early_stopper.early_stop(val_loss):
                    print("Early stopping triggered.")
                    break
        """
    if self.mode == "maximize":
      if metric > self.max_val_metric:
        torch.save(obj = model.state_dict(),
                   f = path)
        print(Fore.BLACK, Back.GREEN + "[INFO]: Model Saved!" + Style.RESET_ALL)

    elif self.mode == "minimize":
      if metric < self.min_val_metric:
        torch.save(obj = model.state_dict(),
                   f = path)
        print(Fore.BLACK, Back.GREEN + "[INFO]: Model Saved!" + Style.RESET_ALL)
