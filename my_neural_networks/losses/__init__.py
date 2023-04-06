"""Loss functions for neural networks."""

# Import the loss functions
from loss import Loss
from categorical_cross_entropy import CategoricalCrossEntropy  # noqa: E501
from binary_cross_entropy import BinaryCrossEntropy
from mse import MSE

# Define the __all__ variable
__all__ = [Loss,
           CategoricalCrossEntropy,
           BinaryCrossEntropy,
           MSE
           ]
