from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """Abstract class for activation functions.

    Activation functions are functions with known derivatives that are used to
    transform the output of a layer.

    Attributes:
        num_inputs (int): Number of inputs to the activation function.
        num_outputs (int): Number of outputs from the activation function.
    """
    @abstractmethod
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """Constructor.

        Args:
            num_inputs (int): Number of inputs to the activation function.
            num_outputs (int): Number of outputs from the activation function.
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the activation function.

        Args:
            x (np.ndarray): Input to the activation function.

        Returns:
            np.ndarray: Output of the activation function.
        """
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass
