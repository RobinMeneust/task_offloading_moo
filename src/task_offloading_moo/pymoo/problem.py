"""Pymoo problem class for the task offloading problem."""

import numpy as np
from pymoo.core.problem import Problem
from task_offloading_moo.data.dataset_gen import Dataset


class TaskOffloadingProblem(Problem):
    """Pymoo problem class for the task offloading problem.

    Attributes:
        dataset_generator (Dataset): Dataset object to generate the data.
    """

    def __init__(self, num_cloud_machines=3, num_fog_machines=10, num_tasks=50, seed=2025):
        """Initialize the problem with the given parameters.

        Args:
            num_cloud_machines (int): Number of cloud machines.
            num_fog_machines (int): Number of fog machines.
            num_tasks (int): Number of tasks.
            seed (int): Random seed for the dataset generator.
        """
        self.dataset_generator = Dataset(
            num_cloud_machines=num_cloud_machines, num_fog_machines=num_fog_machines, num_tasks=num_tasks, seed=seed
        )
        super().__init__(
            n_var=num_tasks, n_obj=Dataset.get_num_objectives(), xl=0, xu=num_cloud_machines + num_fog_machines - 1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the population with the dataset generator.

        Args:
            x (np.ndarray): Population to evaluate.
            out (dict): Dictionary to store the results.
        """
        out["F"] = self.dataset_generator.evaluate_population(x)

    def soft_repair(self, previous_individual, new_individual):
        """Alternative version for repair.

        Consider previous solution and move it towards new one considering boundaries.
        If it's out of bounds after different repairs attempt, it will be set to the previous solution.

        Args:
            previous_individual (np.ndarray): Previous individual.
            new_individual (np.ndarray): New individual.

        Returns:
            np.ndarray: Repaired individual
        """
        return self.dataset_generator.repair_individual_soft(previous_individual, new_individual)


if __name__ == "__main__":
    problem = TaskOffloadingProblem(3, 10, 50)
    pop_size = 3
    x = np.random.randint(0, 13, size=(pop_size, 50))
    F = problem.evaluate(x)
    print(f"Population of size {pop_size} evaluated with {problem.n_obj} objectives: {F}")
