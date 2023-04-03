from my_neural_networks.activations.activation import Activation
import numpy as np


class Tanh(Activation):
    """Tanh activation function.

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
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the activation function.

        The gradient is calculated with respect to the input to the activation
        function. This is used in the backward pass of the neural network.

        Args:
            x (np.ndarray): Input to the activation function.

        Returns:
            np.ndarray: Gradient of the activation function.
        """
        return 1 - np.tanh(x) ** 2
