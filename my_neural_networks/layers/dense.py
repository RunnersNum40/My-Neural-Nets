from my_neural_networks.layers.layer import Layer
from my_neural_networks.initializers.initializer import Initializer
from my_neural_networks.optimizers import Optimizer
import numpy as np


class Dense(Layer):
    """Class for dense layer.

    This class implements a dense layer. A dense layer is a layer where each
    unit is connected to all the units in the previous layer. The forward pass
    of a dense layer is given by:

    y = x * W + b

    where x is the input to the layer, W is the weights of the layer and b is
    the bias of the layer.

    Attributes:
        weights (np.ndarray): Weights of the layer.
        bias (np.ndarray): Bias of the layer.
    """
    def __init__(self,
                 input_shape: np.ndarray,
                 output_shape: np.ndarray,
                 initalizer: Initializer,
                 optimizer: Optimizer
                 ) -> None:
        """Constructor.

        Args:
            input_shape (np.ndarray): Shape of the input to the layer.
            output_shape (np.ndarray): Shape of the output from the layer.
            initalizer (Initializer): Initializer for the parameters of the
                layer.
            optimizer (Optimizer): Optimizer for the parameters of the layer.
        """
        # Call the constructor of the parent class.
        super().__init__(input_shape, output_shape, initalizer, optimizer)
        # Initialize the weights and bias.
        self.weights = self.initializer(self)
        self.bias = np.zeros((1, output_shape))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the layer.

        Args:
            x (np.ndarray): Input to the layer.

        Returns:
            np.ndarray: Output of the layer.
        """
        # Save the input to the layer.
        self.last_input = x
        # Calculate the output of the layer.
        self.last_output = np.dot(x, self.weights) + self.bias
        # Return the output of the layer.
        return self.last_output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass of the layer.

        Args:
            gradient (np.ndarray): Gradient of the loss with respect to the
                output of the layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the
                layer.
        """
        # Save the gradient of the parameters of the layer.
        self.gradient = gradient
        # Calculate the gradient of the loss with respect to the input of the
        # layer.
        gradient = np.dot(gradient, self.weights.T)
        # Return the gradient of the loss with respect to the input of the
        # layer.
        return gradient
