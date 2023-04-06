from typing import List, Callable
import numpy as np

from ..layers import Layer
from ..utils import CompileError, CompiledError
from ..losses import Loss


class Network:
    """Abstract class for neural networks.

    Neural networks are composed of layers. They are used to transform the
    input to the neural network into the output of the neural network.

    Attributes:
        layers (list[Layer]): List of layers in the neural network.
        compiled (bool): Whether the neural network has been compiled.
    """
    def __init__(self, layers: List[Layer]) -> None:
        """Constructor.

        Args:
            layers (list): List of layers in the neural network.
        """
        # Save the layers.
        self.layers = layers

        # Initialize the compiled attribute.
        self.compiled = False

    @property
    def num_inputs(self) -> int:
        """Number of inputs to the neural network.

        Returns:
            int: Number of inputs to the neural network.
        """
        return self.layers[0].num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of outputs from the neural network.

        Returns:
            int: Number of outputs from the neural network.
        """
        return self.layers[-1].num_outputs

    def compile(self) -> None:
        """Compile the neural network.

        This method should be called before training or evaluating the
        neural network.

        Raises:
            CompileError: If the neural network cannot be compiled.
        """
        # Check if the neural network can be compiled.
        for n, (prev_layer, next_layer) in enumerate(zip(self.layers[:-1], self.layers[1:])):  # noqa: E501
            if prev_layer.num_outputs != next_layer.num_inputs:
                raise CompileError(f"Layer {n} has {prev_layer.num_outputs} "
                                   f"outputs but layer {n + 1} has "
                                   f"{next_layer.num_inputs} inputs.")

        # Initalize the layers.
        for layer in self.layers:
            layer.initialize()

        # Set the compiled attribute to True.
        self.compiled = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the neural network.

        Args:
            x (np.ndarray): Input to the neural network.

        Returns:
            np.ndarray: Output of the neural network.
        """
        # Check if the neural network has been compiled.
        if not self.compiled:
            raise CompiledError("The neural network has not been compiled.")

    def update(self) -> None:
        """Update the parameters of the neural network."""
        pass

    def train(self,
              x: np.ndarray,
              y: np.ndarray,
              loss: Loss,
              epochs: int,
              verbose: bool = True,
              on_epoch: Callable[["Network"], None] = None
              ) -> None:
        """Train the neural network.

        Args:
            x (np.ndarray): Input to the neural network.
            y (np.ndarray): Output of the neural network.
            loss (Loss): Loss function.
            epochs (int): Number of epochs to train for.
            verbose (bool): Whether to print the loss after each epoch.
            on_epoch (Callable[["Network"], None]): Function to call after
                each epoch.
        """
        # Check if the neural network has been compiled.
        if not self.compiled:
            raise CompiledError("The neural network has not been compiled.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the neural network.

        Args:
            x (np.ndarray): Inputs to the neural network.

        Returns:
            np.ndarray: Output of the neural network.
        """
        # Check if the neural network has been compiled.
        if not self.compiled:
            raise CompiledError("The neural network has not been compiled.")

        # Forward pass.

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the neural network.

        Args:
            x (np.ndarray): Input to the neural network.
            y (np.ndarray): Output of the neural network.

        Returns:
            float: Loss of the neural network.
        """
        # Check if the neural network has been compiled.
        if not self.compiled:
            raise CompiledError("The neural network has not been compiled.")

    def save(self, path: str) -> None:
        """Save the neural network.

        Args:
            path (str): Path to save the neural network.
        """
        # Check if the neural network has been compiled.
        if not self.compiled:
            raise CompiledError("The neural network has not been compiled.")

        raise NotImplementedError  # TODO: Implement this method.

    @classmethod
    def load(cls, path: str) -> "Network":
        """Load the neural network.

        Args:
            path (str): Path to load the neural network.

        Returns:
            Network: Neural network.
        """
        raise NotImplementedError  # TODO: Implement this method.
