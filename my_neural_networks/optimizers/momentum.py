from my_neural_networks.optimizers.optimizer import Optimizer
import numpy as np


class Momentum(Optimizer):
    """Momentum optimizer.

    Attributes:
        learning_rate (float): Learning rate of the optimizer.
        momentum (float): Momentum of the optimizer.
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
        # If the velocity is None, initialize it to zero.
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        # Update the velocity.
        self.velocity = (self.momentum * self.velocity +
                         self.learning_rate * self.gradient)
        # Return the updated parameters.
        return params - self.velocity
