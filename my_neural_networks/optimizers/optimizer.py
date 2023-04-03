from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """Abstract class for optimizers.

    Optimizers are used to update the parameters of the neural network during
    training.

    Attributes:
        gradient (np.ndarray): Gradient of the parameters of the layer in the
            last backward pass.
    """
    def __init__(self) -> None:
        # Initialize the gradient to None.
        self.gradient = None

    @abstractmethod
    def record(self, gradient: np.ndarray) -> None:
        """Record the gradient of the parameters of the layer.

        Args:
            gradient (np.ndarray): Gradient of the parameters of the layer.
        """
        pass

    @abstractmethod
    def update(self, parameters: np.ndarray) -> np.ndarray:
        """Update the parameters of the layer.

        Args:
            parameters (np.ndarray): Parameters of the layer.

        Returns:
            np.ndarray: Updated parameters of the layer.
        """
        pass
