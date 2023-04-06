from optimizer import Optimizer
from typing import List, Tuple
import numpy as np


class Momentum(Optimizer):
    """Momentum optimizer.

    Attributes:
        learning_rate (float): Learning rate of the optimizer.
        momentum (float): Momentum of the optimizer.
        velocity (np.ndarray): Velocity of the optimizer.
        gradients List[Tuple[np.ndarray, np.ndarray]]: Gradients of
            the weights and biases of the neural network.
    """
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.95
                 ) -> None:
        """Constructor.

        Args:
            learning_rate (float): Learning rate of the optimizer.
            momentum (float): Momentum of the optimizer.
        """
        # Call the constructor of the parent class.
        super().__init__()
        # Save the learning rate and momentum.
        self.learning_rate = learning_rate
        self.momentum = momentum
        # Initialize the velocity to None.
        self.velocity = None

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
        # Calculate the velocity.
        if self.velocity is None:
            self.velocity = (mean_weight_gradient, mean_bias_gradient)
        else:
            self.velocity = (self.momentum * self.velocity[0] + (1 - self.momentum) * mean_weight_gradient,  # noqa: E501
                             self.momentum * self.velocity[1] + (1 - self.momentum) * mean_bias_gradient)  # noqa: E501
        # Calculate the updated weights.
        updated_weights = weights - self.learning_rate * self.velocity[0]
        # Calculate the updated bias.
        updated_bias = bias - self.learning_rate * self.velocity[1]
        # Return the updated weights and bias.
        return updated_weights, updated_bias
