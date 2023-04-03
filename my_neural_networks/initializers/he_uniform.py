from my_neural_networks.initializers.initializer import Initializer
from my_neural_networks.layers import Layer
import numpy as np


class HeUniform(Initializer):
    """Class for He uniform initializer.

    This class implements the He uniform initializer. The weights are
    initialized randomly from a uniform distribution with the specified lower
    and upper bounds.

    Attributes:
        limit (float): Limit of the uniform distribution.
    """
    def __call__(self, layer: Layer) -> np.ndarray:
        """Initialize the weights of a layer.

        Args:
            layer (Layer): Layer to be initialized.

        Returns:
            np.ndarray: Initialized weights.
        """
        # Calculate the number of inputs and outputs.
        num_inputs = np.prod(layer.input_shape)
        num_outputs = np.prod(layer.output_shape)
        # Calculate the limit of the uniform distribution.
        limit = np.sqrt(6.0 / (num_inputs + num_outputs))
        # Calculate the shape of the weights.
        shape = (num_inputs, num_outputs)
        # Return the initialized weights.
        return np.random.uniform(-limit, limit, shape)
