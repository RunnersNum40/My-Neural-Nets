from my_neural_networks.optimizers.optimizer import Optimizer
import numpy as np


class GradientDescent(Optimizer):
    """Gradient descent optimizer.

    Attributes:
        learning_rate (float): Learning rate of the optimizer.
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

    def record(self, gradient: np.ndarray) -> None:
        """Record the gradient of the parameters of the neural network.

        Args:
            gradient (np.ndarray): Gradient of the parameters of the neural
                network.
        """
        # Save the gradient.
        self.gradient = gradient

    def update(self, params: np.ndarray) -> np.ndarray:
        """Update the parameters of the neural network.

        Args:
            params (np.ndarray): Parameters of the neural network.

        Returns:
            np.ndarray: Updated parameters of the neural network.
        """
        # Return the updated parameters.
        return params - self.learning_rate * self.gradient
