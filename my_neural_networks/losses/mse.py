from my_neural_networks.losses.loss import Loss
import numpy as np


class MSE(Loss):
    """Class for mean squared error loss function.

    This class implements the mean squared error loss function. The mean
    squared error is calculated as the mean of the squared differences
    between the true and predicted labels.
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
        loss = np.mean(np.square(y_true - y_pred))
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
        return -2.0 * (y_true - y_pred) / y_true.size
