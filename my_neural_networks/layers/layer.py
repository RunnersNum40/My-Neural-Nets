from abc import ABC, abstractmethod
import numpy as np

from ..optimizers import Optimizer
from ..initializers import Initializer


class Layer(ABC):
    """Abstract class for layers.

    Layers are the building blocks of neural networks. They are used to
    transform the input to the neural network into the output of the neural
    network.

    Attributes:
        input_shape (np.ndarray): Shape of the input to the layer.
        output_shape (np.ndarray): Shape of the output from the layer.
        last_input (np.ndarray): Input to the layer in the last
            forward pass.
        last_output (np.ndarray): Output of the layer in the last
            forward pass.
        gradient (np.ndarray): Gradient of parameters of the layer in the last
            backward pass.
    """
    def __init__(self,
                 input_shape: np.ndarray,
                 output_shape: np.ndarray,
                 initalizer: Initializer,
                 optimizer: Optimizer
                 ) -> None:
        """Constructor.

        Args:
            num_inputs (int): Number of inputs to the layer.
            num_outputs (int): Number of outputs from the layer.
            initalizer (Initializer): Initializer for the parameters of the
                layer.
            optimizer (Optimizer): Optimizer for the parameters of the layer.
        """
        # Save input and output shapes.
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Save the initializer and optimizer.
        self.initializer = initalizer
        self.optimizer = optimizer

        # Initialize attributes to None.
        self.last_input = None
        self.last_output = None
        self.gradient = None

        # Initialize the parameters of the layer.
        self.weights = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the layer.

        Args:
            x (np.ndarray): Input to the layer.

        Returns:
            np.ndarray: Output of the layer.
        """
        pass

    @abstractmethod
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass of the layer.

        Args:
            gradient (np.ndarray): Gradient of the loss with respect to the
                output of the layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the
                layer.
        """
        pass

    def initalize(self) -> None:
        """Initialize the parameters of the layer."""
        self.weights = self.initializer(self)

    def optimize(self) -> None:
        """Optimize the parameters of the layer."""
        # Update the parameters of the layer.
        self.weights = self.optimizer.update(self.weights)
