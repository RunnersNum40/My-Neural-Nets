from my_neural_networks.initializers.initializer import Initializer
from my_neural_networks.layers import Layer
from typing import Tuple
import numpy as np


class HeUniform(Initializer):
    """Class for He uniform initializer.

    This class implements the He uniform initializer. The weights are
    initialized randomly from a uniform distribution with the specified lower
    and upper bounds.

    Attributes:
        limit (float): Limit of the uniform distribution.
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
        # Calculate the limit of the uniform distribution.
        limit = np.sqrt(6.0 / (num_inputs + num_outputs))
        # Calculate the initialized weights.
        weights = np.random.uniform(-limit, limit, (num_inputs, num_outputs))
        # Calculate the initialized bias.
        bias = np.random.uniform(-limit, limit, (1, num_outputs))
        # Return the initialized weights and bias.
        return weights, bias
