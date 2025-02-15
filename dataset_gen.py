# based on https://www.kaggle.com/datasets/sachin26240/vehicularfogcomputing
# distance: https://ecssria.eu/assets/images/table_latency.png
from idlelib.pyparse import trans

import numpy as np


class Dataset:
    def __init__(self, num_cloud_machines, num_fog_machines, num_tasks, values_domains=None):
        self._propagation_speed = 3e8/1.44  # m/s (https://www.rp-photonics.com/fibers.html)

        if values_domains is None:
            values_domains = {
                "cloud_machine": {
                    "cpu_rate": [3000, 5000],  # MIPS (1e6)
                    "cpu_usage_cost": [0.7/3600, 1.0/3600],  # €/s
                    "ram_usage_cost": [0.02/3600, 0.05/3600],  # €/GB/s
                    "bandwidth_usage_cost": [0.05, 0.1],  # €/GB
                    "latency": [0.01, 3],  # s
                    "distance_edge": [1e4, 1e6],  # m
                    "bandwidth": [10, 400],  # GB/s
                    "ram_limit": [64, 512],  # GB
                },
                "fog_machine": {
                    "cpu_rate": [500, 1500],  # MIPS (1e6)
                    "cpu_usage_cost": [0.1/3600, 0.4/3600],  # €/s
                    "ram_usage_cost": [0.01/3600, 0.03/3600],  # €/GB/s
                    "bandwidth_usage_cost": [0.01, 0.02],  # €/GB
                    "latency": [0.005, 0.01],  # s
                    "distance_edge": [1e2, 5e4],  # m
                    "bandwidth": [0.05, 10],  # GB/s
                    "ram_limit": [8, 32],  # GB
                },
                "task": {
                    "num_instructions": [1000, 1000],  # millions of instructions (1e6)
                    "ram_required": [0.05, 0.2],  # GB
                    "in_traffic": [0.01, 0.1],  # GB
                    "out_traffic": [0.01, 0.1],  # GB
                }
            }

            # edge_node ?

        cloud_machines = self.gen_machines(num_cloud_machines, values_domains["cloud_machine"])
        fog_machines = self.gen_machines(num_fog_machines, values_domains["fog_machine"])
        self._machines = np.concatenate((cloud_machines, fog_machines))
        self._tasks = self.gen_tasks(num_tasks, values_domains["task"])

        # pre-compute objective values
        self._run_time_matrix, self._cost_matrix = self._compute_run_time_cost_matrices()

    def _compute_latency(self, distance, data_quantity, bandwidth):
        propagation_time = 2 * (distance / self._propagation_speed) # *2 because it goes and comes back
        transfer_time = data_quantity / bandwidth # data_quantity = in & out

        return propagation_time + transfer_time

    @staticmethod
    def _compute_execution_time(task, machine):
        return task["num_instructions"] / machine["cpu_rate"]

    @staticmethod
    def _compute_cost(task, machine, exec_time):
        return (machine["cpu_usage_cost"] + machine["ram_usage_cost"] * task["ram_required"]) * exec_time + machine["bandwidth_usage_cost"] * (task["in_traffic"] + task["out_traffic"])

    def _compute_total_time(self, task, machine, exec_time):
        return exec_time + self._compute_latency(machine["distance_edge"], task["in_traffic"] + task["out_traffic"], machine["bandwidth"])

    def _compute_run_time_cost_matrices(self):
        run_time_matrix = np.zeros((self.get_num_tasks(), self.get_num_machines()))
        cost_matrix = np.zeros((self.get_num_tasks(), self.get_num_machines()))

        for i, task in enumerate(self.get_tasks()):
            for j, machine in enumerate(self.get_machines()):
                exec_time = self._compute_execution_time(task, machine)  # used for the 2 objectives
                cost_matrix[i, j] = self._compute_cost(task, machine, exec_time)
                run_time_matrix[i, j] = self._compute_total_time(task, machine, exec_time)

        return run_time_matrix, cost_matrix

    def gen_machines(self, num_machines, variables_domain):
        machines = np.empty(num_machines, dtype=object)
        machines_values = {}
        for k, var_range in variables_domain.items():
            if type(var_range[0]) == int:
                machines_values[k] = np.random.randint(var_range[0], var_range[1] + 1, size=num_machines)
            elif type(var_range[0]) == float:
                machines_values[k] = np.random.rand(num_machines) * (var_range[1] - var_range[0]) + var_range[0]
            else:
                raise Exception("invalid type (should be int or float)")
        for i in range(num_machines):
            machines[i] = {k: machines_values[k][i] for k in machines_values.keys()}
        return machines

    def gen_tasks(self, num_tasks, variables_domain):
        tasks = np.empty(num_tasks, dtype=object)
        tasks_values = {}
        for k, var_range in variables_domain.items():
            if type(var_range[0]) == int:
                tasks_values[k] = np.random.randint(var_range[0], var_range[1] + 1, size=num_tasks)
            elif type(var_range[0]) == float:
                tasks_values[k] = np.random.rand(num_tasks) * (var_range[1] - var_range[0]) + var_range[0]
            else:
                raise Exception("invalid type (should be int or float)")

        for i in range(num_tasks):
            tasks[i] = {k: tasks_values[k][i] for k in tasks_values.keys()}
        return tasks

    def get_tasks(self):
        return self._tasks

    def get_machines(self):
        return self._machines

    def init_pop(self, pop_size):
        pop = np.empty((pop_size, len(self._tasks)), dtype=int)
        for i in range(pop_size):
            pop[i] = np.random.randint(0, len(self._machines), size=len(self._tasks))
        return pop

    def get_num_tasks(self):
        return len(self._tasks)

    def get_num_machines(self):
        return len(self._machines)

    def get_cost(self, task_idx, machine_idx):
        return self._cost_matrix[task_idx, machine_idx]

    def get_run_time(self, task_idx, machine_idx):
        return self._run_time_matrix[task_idx, machine_idx]


# TEST

dataset = Dataset(num_cloud_machines=3, num_fog_machines=10, num_tasks=10)

pop = dataset.init_pop(pop_size=3)
print(pop)

print("task 0:", dataset.get_tasks()[0])
print("machine 0:", dataset.get_machines()[0])

print(f"Cost of task 0 on machine 0: {dataset.get_cost(0, 0)}")
print(f"Time of task 0 on machine 0: {dataset.get_run_time(0, 0)}")



