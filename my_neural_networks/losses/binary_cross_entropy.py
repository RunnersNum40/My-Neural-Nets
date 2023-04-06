from loss import Loss
import numpy as np


class BinaryCrossEntropy(Loss):
    """Class for binary cross entropy loss function.

    This class implements the binary cross entropy loss function. The binary
    cross entropy is calculated as the negative mean of the true labels times
    the natural logarithm of the predicted labels plus the true labels times
    the natural logarithm of one minus the predicted labels.

    The binary cross entropy loss function is used for binary classification
    problems.
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
        loss = -np.mean(y_true * np.log(y_pred) +
                        (1 - y_true) * np.log(1 - y_pred))
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
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
