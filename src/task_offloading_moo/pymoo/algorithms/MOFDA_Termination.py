"""Module for termination criteria for MOFDA algorithm."""

from pymoo.core.termination import Termination


class MOFDATermination(Termination):
    """Termination criteria for MOFDA algorithm."""

    def __init__(self, n_max_gen=100):
        """Initialize the termination criteria with the maximum number of generations."""
        super().__init__()
        self.n_max_gen = n_max_gen

    def _update(self, algorithm):
        # Calculate progress as percentage of maximum generations
        if algorithm.n_gen >= self.n_max_gen:
            return 1.0
        else:
            return algorithm.n_gen / self.n_max_gen
