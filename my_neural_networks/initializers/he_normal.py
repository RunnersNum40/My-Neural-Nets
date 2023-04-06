from initializer import Initializer
from my_neural_networks.layers import Layer
from typing import Tuple
import numpy as np


class HeNormal(Initializer):
    """Class for He normal initializer.

    This class implements the He normal initializer. The weights are
    initialized randomly from a normal distribution with the specified
    mean and standard deviation. The mean is 0 and the standard deviation
    is sqrt(2 / fan_in), where fan_in is the number of input units
    to the layer.
    """
    def __call__(self, layer: Layer) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize the weights of a layer.

        Args:
            layer (Layer): Layer to be initialized.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the weights and
                biases of the layer.
        """
        # Calculate the number of inputs and outputs.
        num_inputs = np.prod(layer.input_shape)
        num_outputs = np.prod(layer.output_shape)
        # Calculate the standard deviation.
        std = np.sqrt(2.0 / num_inputs)
        # Calculate the initialized weights.
        weights = np.random.normal(0.0, std, (num_inputs, num_outputs))
        # Calculate the initialized bias.
        bias = np.random.normal(0.0, std, (1, num_outputs))
        # Return the initialized weights and bias.
        return weights, bias
