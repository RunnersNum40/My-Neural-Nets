from activation import Activation
import numpy as np


class Sigmoid(Activation):
    """Sigmoid activation function.

    Attributes:
        num_inputs (int): Number of inputs to the activation function.
        num_outputs (int): Number of outputs from the activation function.
    """
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """Constructor.

        The number of inputs and outputs is used to determine ensure that the
        activation function is applied to the correct number of inputs and
        outputs.

        Args:
            num_inputs (int): Number of inputs to the activation function.
            num_outputs (int): Number of outputs from the activation function.
        """
        super().__init__(num_inputs, num_outputs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the activation function.

        The forward pass is used in the forward pass of the neural network.

        Args:
            x (np.ndarray): Input to the activation function.

        Returns:
            np.ndarray: Output of the activation function.
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the activation function.

        The gradient is calculated with respect to the input to the activation
        function. This is used in the backward pass of the neural network.

        Args:
            x (np.ndarray): Input to the activation function.

        Returns:
            np.ndarray: Gradient of the activation function.
        """
        return self.forward(x) * (1 - self.forward(x))
