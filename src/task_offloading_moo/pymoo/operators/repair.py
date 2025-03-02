"""This file contains the repair operator for the task offloading problem."""

from pymoo.core.repair import Repair


class TaskOffloadingRepair(Repair):
    """Repair operator for the task offloading problem."""

    def _do(self, problem, X, **kwargs):
        """Repair the population for the task offloading problem.

        Args:
            problem (TaskOffloadingProblem): Problem instance.
            X (np.ndarray): Population to repair.

        Returns:
            np.ndarray: Repaired population.
        """
        X = [problem.dataset_generator.repair_individual(x) for x in X]
        return X
