from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from ..layers import Layer


class Initializer(ABC):
    """Abstract class for initializers.

    Initializers are used to initialize the parameters of a layer.
    """
    @abstractmethod
    def __call__(self, layer: Layer) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize the weights of a layer.

        Args:
            layer (Layer): Layer to be initialized.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the weights and
                biases of the layer.
        """
        pass
