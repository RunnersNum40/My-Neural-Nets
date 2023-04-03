from abc import ABC, abstractmethod
from typing import Tuple
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
    def record(self,
               weight_gradient: np.ndarray,
               bias_gradient: np.ndarray
               ) -> None:
        """Record the gradient of the parameters of the layer.

        Args:
            gradient (np.ndarray): Gradient of the parameters of the layer.
            bias_gradient (np.ndarray): Gradient of the bias of the layer.
        """
        pass

    @abstractmethod
    def update(self,
               weights: np.ndarray,
               bias: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Update the parameters of the layer.

        Args:
            weights (np.ndarray): Parameters of the layer.
            bias (np.ndarray): Bias of the layer.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated weights and bias.
        """
        pass
