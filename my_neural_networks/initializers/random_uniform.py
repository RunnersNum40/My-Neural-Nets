from initializer import Initializer
from my_neural_networks.layers import Layer
from typing import Tuple
import numpy as np


class RandomUniform(Initializer):
    """Class for random uniform initializer.

    This class implements the random uniform initializer. The weights are
    initialized randomly from a uniform distribution with the specified lower
    and upper bounds.

    Attributes:
        lower_bound (float): Lower bound of the uniform distribution.
        upper_bound (float): Upper bound of the uniform distribution.
    """
    def __init__(self,
                 lower_bound: float = -0.05,
                 upper_bound: float = 0.05
                 ) -> None:
        """Constructor.

        Args:
            lower_bound (float): Lower bound of the uniform distribution.
            upper_bound (float): Upper bound of the uniform distribution.
        """
        # Call the constructor of the parent class.
        super().__init__()
        # Save the lower and upper bounds.
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

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
        # Calculate the initialized weights.
        weights = np.random.uniform(self.lower_bound,
                                    self.upper_bound,
                                    (num_inputs, num_outputs)
                                    )
        # Calculate the initialized bias.
        bias = np.random.uniform(self.lower_bound,
                                 self.upper_bound,
                                 (1, num_outputs)
                                 )
        # Return the initialized weights and bias.
        return weights, bias
