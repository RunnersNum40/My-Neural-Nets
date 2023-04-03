# Import all the activation functions
from my_neural_networks.activations.activation import Activation
from my_neural_networks.activations.elu import ELU
from my_neural_networks.activations.lelu import Lelu
from my_neural_networks.activations.linear import Linear
from my_neural_networks.activations.relu import ReLU
from my_neural_networks.activations.sigmoid import Sigmoid
from my_neural_networks.activations.softmax import Softmax
from my_neural_networks.activations.tanh import Tanh


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
