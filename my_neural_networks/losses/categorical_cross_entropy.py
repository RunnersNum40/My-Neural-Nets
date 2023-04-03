from my_neural_networks.losses.loss import Loss
import numpy as np


class CategoricalCrossEntropy(Loss):
    """Class for categorical cross entropy loss function.

    This class implements the categorical cross entropy loss function. The
    categorical cross entropy is calculated as the negative mean of the true
    labels times the natural logarithm of the predicted labels.

    The categorical cross entropy loss function is used for multi-class
    classification problems.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the loss.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Loss.
        """
        # Calculate the loss.
        loss = -np.sum(y_true * np.log(y_pred))
        # Return the loss.
        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the loss.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Gradient of the loss.
        """
        # Return the gradient.
        return -y_true / y_pred
