from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """Abstract class for loss functions.

    Loss functions are used to measure the performance of the neural network
    during training. The loss function is used to calculate the gradient of
    the neural network parameters.
    """
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the loss.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Loss.
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the loss.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Gradient of the loss.
        """
        pass
