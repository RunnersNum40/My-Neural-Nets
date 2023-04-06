class NetworkError(Exception):
    """Base class for exceptions in this module."""
    pass


class CompileError(NetworkError):
    """Exception raised when the neural network cannot be compiled."""
    pass


class CompiledError(NetworkError):
    """Exception raised when the neural network has not been compiled."""
    pass
