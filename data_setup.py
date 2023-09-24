import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader


def fashion_mnist_setup():
  """
    Set up data loaders for the Fashion MNIST dataset.

    This function prepares the Fashion MNIST dataset by splitting it into training and validation sets
    and creating data loaders for both training, validation, and testing data.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the training dataset.
        torch.utils.data.DataLoader: DataLoader for the validation dataset.
        torch.utils.data.DataLoader: DataLoader for the testing dataset.

    Example:
    ```python
    train_loader, val_loader, test_loader = fashion_mnist_setup()
    for batch in train_loader:
        # Training iteration code...
    ```
  """
  mnist_train = FashionMNIST(root = "data",
                           train=True,
                           transform = ToTensor(),
                           download = True)

  mnist_test = FashionMNIST(root = "data",
                            train = False,
                            transform = ToTensor(),
                            download=True)

  train, val = train_test_split(mnist_train,
                                test_size = 0.15,
                                random_state = 42)

  train_loader = DataLoader(train, batch_size = 128, shuffle = True, pin_memory = True)
  val_loader = DataLoader(val, batch_size = 128, shuffle = False, pin_memory = True)
  test_loader = DataLoader(mnist_test, batch_size = 128, shuffle = False, pin_memory = True)

  return train_loader, val_loader, test_loader
