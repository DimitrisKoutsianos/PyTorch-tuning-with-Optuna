import torch
from torch import nn
import optuna
from torch.nn import functional as F

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

  n_layers = trial.suggest_int("n_layers", 1, 5)    #the number of layers the MLP will have
  layers = []    #a list where we will append the layers of the MLP
  input_shape = 28*28    #since our images are 28x28 (HxW), the input shape will be 28*28 (H*W)
  for i in range(n_layers):
      output_shape = trial.suggest_categorical(f"Layer_{i+1}", [2 ** i for i in range(5, 10)])    #the output shape of each layer
      layers.append(nn.Linear(input_shape, output_shape))    #appending the Linear layer to the layers list
      act = trial.suggest_categorical(f"Activation_{i+1}", ["ReLU", "SiLU"])    #the activation function of each layer
      layers.append(getattr(nn, act)())    #appending the activation function
      drop = trial.suggest_categorical(f"Dropout_{i+1}", [0,1])    #deciding if we should add a dropout layer or not
      if drop == 1:   
        layers.append(nn.Dropout(trial.suggest_categorical(f"Dropout_rate_{i+1}", [0.1,0.15,0.2,0.25,0.3,0.4,0.5])))    #adding a dropout layer 

      batchnorm = trial.suggest_categorical(f"BatchNorm_{i+1}", [0,1])  #deciding if we should add a Batch Normalization layer or not
      if batchnorm == 1:
        layers.append(nn.BatchNorm1d(output_shape))    #adding a Batch Normalization layer
      input_shape = output_shape    #turning the output shape to input shape so that the next Linear layer accepts the correct input shape

  layers.append(nn.Linear(output_shape, 10))    #adding a final Linear layer with 10 output nodes, one for each of our classes
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
    return F.log_softmax(self.model(data.view(-1,28*28)))
