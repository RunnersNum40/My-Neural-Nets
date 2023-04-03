from my_neural_networks.optimizers.optimizer import Optimizer
from typing import List, Tuple
import numpy as np


class GradientDescent(Optimizer):
    """Gradient descent optimizer.

    Attributes:
        learning_rate (float): Learning rate of the optimizer.
        gradients List[Tuple[np.ndarray, np.ndarray]]: Gradients of
            the weights and biases of the neural network.
    """
    def __init__(self, learning_rate: float = 0.01) -> None:
        """Constructor.

        Args:
            learning_rate (float): Learning rate of the optimizer.
        """
        # Call the constructor of the parent class.
        super().__init__()
        # Save the learning rate.
        self.learning_rate = learning_rate
        # Initialize the gradients to None.
        self.gradients: List[Tuple[np.ndarray, np.ndarray]] = []

    def record(self,
               weight_gradient: np.ndarray,
               bias_gradient: np.ndarray
               ) -> None:
        """Record the gradient of the parameters of the layer.

        Args:
            gradient (np.ndarray): Gradient of the parameters of the layer.
            bias_gradient (np.ndarray): Gradient of the bias of the layer.
        """
        # Save the gradient.
        self.gradients.append((weight_gradient, bias_gradient))

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
        # Calculate the mean weight gradient.
        mean_weight_gradient = np.mean([gradient[0] for gradient in self.gradients], axis=0)  # noqa: E501
        # Calculate the mean bias gradient.
        mean_bias_gradient = np.mean([gradient[1] for gradient in self.gradients], axis=0)  # noqa: E501
        # Calculate the updated weights.
        updated_weights = weights - self.learning_rate * mean_weight_gradient
        # Calculate the updated bias.
        updated_bias = bias - self.learning_rate * mean_bias_gradient
        # Clear the gradients.
        self.gradients = []
        # Return the updated weights and bias.
        return updated_weights, updated_bias
