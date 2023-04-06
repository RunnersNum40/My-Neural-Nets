from initializer import Initializer, Layer
from typing import Tuple
import numpy as np


class RandomNormal(Initializer):
    """Class for random normal initializer.

    This class implements the random normal initializer. The weights are
    initialized randomly from a normal distribution with the specified mean and
    standard deviation.

    Attributes:
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
    """
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Constructor.

        Args:
            mean (float): Mean of the normal distribution.
            std (float): Standard deviation of the normal distribution.
        """
        # Call the constructor of the parent class.
        super().__init__()
        # Save the mean and standard deviation.
        self.mean = mean
        self.std = std

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
        weights = np.random.normal(self.mean,
                                   self.std,
                                   (num_inputs, num_outputs)
                                   )
        # Calculate the initialized bias.
        bias = np.random.normal(self.mean, self.std, (1, num_outputs))
        # Return the initialized weights and bias.
        return weights, bias
