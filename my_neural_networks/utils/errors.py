class CompileError(Exception):
    """Exception raised when the neural network cannot be compiled."""
    pass


class CompiledError(Exception):
    """Exception raised when the neural network has not been compiled."""
    pass
