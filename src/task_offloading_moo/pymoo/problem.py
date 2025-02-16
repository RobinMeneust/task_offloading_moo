import numpy as np
from pymoo.core.problem import Problem
from task_offloading_moo.data.dataset_gen import Dataset


class TaskOffloadingProblem(Problem):
    def __init__(self, num_cloud_machines=3, num_fog_machines=10, num_tasks=50, seed=2025):
        self.dataset_generator = Dataset(num_cloud_machines=num_cloud_machines, num_fog_machines=num_fog_machines, num_tasks=num_tasks, seed=seed)
        super().__init__(n_var=num_tasks, n_obj=Dataset.get_num_objectives(), xl=0, xu=num_cloud_machines + num_fog_machines - 1)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [self.dataset_generator.get_objectives(x[i]) for i in range(len(x))]


if __name__ == "__main__":
    problem = TaskOffloadingProblem(3, 10, 50)
    pop_size = 3
    x = np.random.randint(0, 13, size=(pop_size, 50))
    F = problem.evaluate(x)
    print(f"Population of size {pop_size} evaluated with {problem.n_obj} objectives: {F}")
