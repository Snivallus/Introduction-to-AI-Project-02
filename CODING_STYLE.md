# Code Style Guide

This project follows a consistent coding style for maintainability and clarity.

## Language

All code, comments, and documentation are written in **English**.

## Python Style

Follow [PEP 8](https://pep8.org/) for Python code style. Use 4 spaces per indentation level, maximum line length of 88 characters (Black-compatible).

## Type Annotations

Use Python's type hints for all function signatures and class methods:

```python
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from torch import nn
import numpy as np

def example_function(
    data: torch.Tensor,
    labels: torch.Tensor,
    config: Dict[str, Any],
    batch_size: int = 32
) -> Tuple[torch.Tensor, float]:
    """Example function with type annotations."""
    ...
```

## Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions, classes, and modules.

### Function/Method Docstring Template

```python
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """Train a neural network model.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for parameter updates.
        num_epochs: Number of training epochs.
        device: Device to train on ("cuda" or "cpu").

    Returns:
        Dictionary containing:
            - 'train_losses': List of average training loss per epoch.
            - 'train_accuracies': List of training accuracy per epoch.

    Raises:
        ValueError: If device is not "cuda" or "cpu".
        RuntimeError: If CUDA is requested but not available.

    Example:
        >>> losses = train_model(net, trainloader, criterion, optimizer, 10)
        >>> print(losses['train_losses'][-1])
        0.123
    """
    ...
```

### Class Docstring Template

```python
class CNNClassifier(nn.Module):
    """Convolutional Neural Network for image classification.

    This network follows a LeNet-style architecture with two convolutional
    layers and three fully connected layers. Designed for 32×32 RGB images.

    Attributes:
        conv1: First convolutional layer (3→6 channels, 5×5 kernel).
        conv2: Second convolutional layer (6→16 channels, 5×5 kernel).
        fc1: First fully connected layer (400→120 units).
        fc2: Second fully connected layer (120→84 units).
        fc3: Output layer (84→10 units, one per CIFAR-10 class).

    Example:
        >>> model = CNNClassifier()
        >>> x = torch.randn(4, 3, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([4, 10])
    """

    def __init__(self, dropout_rate: float = 0.5):
        """Initialize CNNClassifier.

        Args:
            dropout_rate: Dropout probability for regularization.
        """
        super().__init__()
        ...
```

## Notebook-Specific Guidelines

For Jupyter notebooks (`project.ipynb`):

1. **Markdown cells**: Use English for explanations.
2. **Code cells**: Follow the same style as Python modules.
3. **Imports**: Group imports at the top of each relevant section:
   ```python
   # Standard library
   import os
   from typing import Dict, List
   
   # Third-party
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torchvision import datasets, transforms
   
   # Visualization
   import matplotlib.pyplot as plt
   ```
4. **Variable names**: Use descriptive names (`train_loader` not `tl`, `num_epochs` not `ep`).
5. **Magic commands**: Place `%matplotlib inline` and similar commands in their own cell.

## PyTorch-Specific Conventions

1. **Device handling**: Explicitly move tensors and models:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   inputs, labels = inputs.to(device), labels.to(device)
   ```

2. **Training loops**: Structure clearly:
   ```python
   model.train()
   for epoch in range(num_epochs):
       running_loss = 0.0
       
       for batch_idx, (inputs, labels) in enumerate(train_loader):
           inputs, labels = inputs.to(device), labels.to(device)
           
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           
           running_loss += loss.item()
           
       avg_loss = running_loss / len(train_loader)
       print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
   ```

3. **Model definitions**: Use `nn.Module` subclassing with clear forward method:
   ```python
   class SimpleCNN(nn.Module):
       """Simple CNN for CIFAR-10."""
       
       def __init__(self, num_classes: int = 10):
           """Initialize SimpleCNN.
           
           Args:
               num_classes: Number of output classes (10 for CIFAR-10).
           """
           super().__init__()
           self.features = nn.Sequential(
               nn.Conv2d(3, 16, kernel_size=3, padding=1),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(2),
               # ... more layers
           )
           self.classifier = nn.Sequential(
               nn.Linear(16 * 16 * 16, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes)
           )
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass.
           
           Args:
               x: Input tensor of shape (batch, 3, 32, 32).
               
           Returns:
               Output tensor of shape (batch, num_classes).
           """
           x = self.features(x)
           x = torch.flatten(x, 1)
           x = self.classifier(x)
           return x
   ```

## File Organization

When creating Python modules:

1. **Module-level docstring**: Start each `.py` file with a module docstring.
2. **Imports**: Group and order imports as shown above.
3. **Constants**: Use uppercase with underscores for constants.
4. **Private functions**: Prefix with underscore `_` for internal helpers.

*Note: This style guide is intended for both notebook development and potential refactoring into Python modules. Following these conventions will make the code more readable, maintainable, and professional.*