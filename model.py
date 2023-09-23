import torch
from torch import nn
import optuna

def suggest_architecture(trial):
  """
    Generate a list of layers for a multi-layer perceptron (MLP) architecture based on the provided Optuna trial.

    Parameters:
        trial (optuna.Trial): The Optuna trial object to suggest hyperparameters from.

    Returns:
        list: A list of PyTorch nn.Module layers representing the architecture.

    Example:
    ```python
    import torch.nn as nn
    import optuna

    # Create an Optuna study
    study = optuna.create_study(direction='maximize')

    # Define a function to optimize
    def objective(trial):
        layers = suggest_architecture(trial)
        model = MLP(layers)
        # ... Add training and evaluation code here ...
        return accuracy

    study.optimize(objective, n_trials=100)
    ```
  """
  n_layers = trial.suggest_int("n_layers", 1, 5)
  layers = []
  input_shape = 28*28
  for i in range(n_layers):
      output_shape = trial.suggest_categorical(f"Layer_{i+1}", [2 ** i for i in range(5, 10)])
      layers.append(nn.Linear(input_shape, output_shape))
      act = trial.suggest_categorical(f"Activation_{i+1}", ["ReLU", "SiLU"])
      layers.append(getattr(nn, act)())
      drop = trial.suggest_categorical(f"Dropout_{i+1}", [0,1])
      if drop == 1:
        layers.append(nn.Dropout(trial.suggest_categorical(f"Dropout_rate_{i+1}", [0.1,0.15,0.2,0.25,0.3,0.4,0.5])))

      batchnorm = trial.suggest_categorical(f"BatchNorm_{i+1}", [0,1])
      if batchnorm == 1:
        layers.append(nn.BatchNorm1d(output_shape))
      input_shape = output_shape

  layers.append(nn.Linear(output_shape, 10))
  return layers

class MLP(nn.Module):
  """
    A multi-layer perceptron (MLP) neural network model.

    Parameters:
        layers (list): A list of PyTorch nn.Module layers representing the network architecture.

    Example:
    ```python
    layers = suggest_architecture(optuna.create_trial())
    model = MLP(layers)
    ```
  """
  def __init__(self, layers):
    super().__init__()

    self.model = nn.Sequential(*layers)

  def forward(self, data):
    """
    Forward pass of the MLP.

    Parameters:
        data (torch.Tensor): Input data of shape (batch_size, input_dim).

    Returns:
        torch.Tensor: Output predictions of shape (batch_size, num_classes).
    """
    return self.model(data.view(-1,28*28))
