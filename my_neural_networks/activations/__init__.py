"""Activation functions for neural networks."""

# Import all the activation functions
from activation import Activation
from elu import ELU
from lelu import Lelu
from linear import Linear
from relu import ReLU
from sigmoid import Sigmoid
from softmax import Softmax
from tanh import Tanh


# Define the __all__ variable
__all__ = [Activation,
           ELU,
           Lelu,
           Linear,
           ReLU,
           Sigmoid,
           Softmax,
           Tanh
           ]
