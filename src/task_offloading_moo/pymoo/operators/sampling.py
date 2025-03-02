"""This file contains the sampling operator for the task offloading problem."""

from pymoo.core.sampling import Sampling


class TaskOffloadingSampling(Sampling):
    """Sampling operator for the task offloading problem."""

    def _do(self, problem, n_samples, **kwargs):
        """Create a population of samples for the task offloading problem.

        Args:
            problem (TaskOffloadingProblem): Problem instance.
            n_samples (int): Number of samples to create.

        Returns:
            np.ndarray: Population of samples.
        """
        return problem.dataset_generator.create_pop(n_samples)
