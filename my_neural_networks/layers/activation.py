from layer import Layer
from my_neural_networks.activations.activation import Activation
import numpy as np


class Activation(Layer):
    """Activation layer.

    Activation layers are used to apply an activation function to the output of
    a layer. They introduce non-linearity to neural networks.

    Attributes:
        activation (Activation): Activation function.
        num_inputs (int): Number of inputs to the activation function.
        num_outputs (int): Number of outputs from the activation function.
        last_input (np.ndarray): Input to the activation layer in the last
            forward pass.
        last_output (np.ndarray): Output of the activation layer in the last
            forward pass.
        gradient (np.ndarray): Gradient of parameters of the activation layer
            in the last backward pass.
    """
    def __init__(self,
                 activation: Activation,
                 num_inputs: int,
                 num_outputs: int
                 ) -> None:
        """Constructor.

        Args:
            activation (Activation): Activation function.
            num_inputs (int): Number of inputs to the activation function.
            num_outputs (int): Number of outputs from the activation function.
        """
        # Call the constructor of the parent class.
        super().__init__(num_inputs, num_outputs)
        # Save the activation function.
        self.activation = activation

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the activation layer.

        The forward pass is used in the forward pass of the neural network.

        Args:
            x (np.ndarray): Input to the activation layer.

        Returns:
            np.ndarray: Output of the activation layer.
        """
        return self.activation.forward(x)

    def _backward(self, grad: np.ndarray) -> np.ndarray:
        """Gradient of the activation layer.

        The gradient is calculated with respect to the input to the activation
        layer. This is used in the backward pass of the neural network.

        Args:
            grad (np.ndarray): Gradient of the loss with respect to the output
                of the activation layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input to the
                activation layer.
        """
        return self.activation.backward(grad)
