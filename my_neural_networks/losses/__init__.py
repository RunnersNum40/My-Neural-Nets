# Import the loss functions
from my_neural_networks.losses.loss import Loss
from my_neural_networks.losses.categorical_cross_entropy import CategoricalCrossEntropy  # noqa: E501
from my_neural_networks.losses.binary_cross_entropy import BinaryCrossEntropy
from my_neural_networks.losses.mse import MSE

# Define the __all__ variable
__all__ = [Loss,
           CategoricalCrossEntropy,
           BinaryCrossEntropy,
           MSE
           ]
