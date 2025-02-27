# based on https://www.kaggle.com/datasets/sachin26240/vehicularfogcomputing
# https://aws.amazon.com/ec2/pricing/on-demand/
# https://instances.vantage.sh/aws/ec2/c4.4xlarge
# cloud: ~= c4.large -> c4.xlarge
# fog: ~= t3.small -> t3.xlarge
# distance: https://ecssria.eu/assets/images/table_latency.png

import numpy as np

from task_offloading_moo.data.load_data_config import load_data_config


class Dataset:
    def __init__(self, num_cloud_machines, num_fog_machines, num_tasks, use_random_machines=True, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._propagation_speed = 3e8 / 1.44  # m/s (https://www.rp-photonics.com/fibers.html)

        tasks_domains = load_data_config("tasks_domains")
        self._tasks = self.gen_items_from_domain(num_tasks, tasks_domains)

        if use_random_machines:
            machines_domains = load_data_config("machines_domains")
            cloud_machines = self.gen_items_from_domain(
                num_cloud_machines, machines_domains["cloud"], same_scale_pos=True
            )
            fog_machines = self.gen_items_from_domain(num_fog_machines, machines_domains["fog"], same_scale_pos=True)
        else:
            raise Exception("Not yet implemented")
            # machines_lists = load_data_config("machines_lists")
            # cloud_machines = self.gen_items_from_list(num_cloud_machines, machines_lists["cloud"])
            # fog_machines = self.gen_items_from_list(num_fog_machines, machines_lists["fog"])

        self._machines = np.concatenate((cloud_machines, fog_machines))

        # pre-compute objective values
        self._run_time_matrix, self._cost_matrix = self._compute_run_time_cost_matrices()

    def _compute_latency(self, distance, data_quantity, bandwidth):
        propagation_time = 2 * (distance / self._propagation_speed)  # *2 because it goes and comes back
        transfer_time = data_quantity / bandwidth  # data_quantity = in & out

        return propagation_time + transfer_time

    @staticmethod
    def _compute_execution_time(task, machine):
        return task["num_instructions"] / machine["cpu_rate"]

    @staticmethod
    def _compute_cost(task, machine, exec_time):
        return (machine["cpu_usage_cost"] + machine["ram_usage_cost"] * task["ram_required"]) * exec_time + machine[
            "bandwidth_usage_cost"
        ] * (task["in_traffic"] + task["out_traffic"])

    def _compute_total_time(self, task, machine, exec_time):
        return exec_time + self._compute_latency(
            machine["distance_edge"], task["in_traffic"] + task["out_traffic"], machine["bandwidth"]
        )

    def _compute_run_time_cost_matrices(self):
        run_time_matrix = np.zeros((self.get_num_tasks(), self.get_num_machines()))
        cost_matrix = np.zeros((self.get_num_tasks(), self.get_num_machines()))

        for i, task in enumerate(self.get_tasks()):
            for j, machine in enumerate(self.get_machines()):
                exec_time = self._compute_execution_time(task, machine)  # used for the 2 objectives
                cost_matrix[i, j] = self._compute_cost(task, machine, exec_time)
                run_time_matrix[i, j] = self._compute_total_time(task, machine, exec_time)

        return run_time_matrix, cost_matrix

    def _is_ram_enough(self, task_idx, machine_idx):
        if (
            task_idx < 0
            or task_idx >= len(self.get_tasks())
            or machine_idx < 0
            or machine_idx >= len(self.get_machines())
        ):
            raise IndexError("task_idx or machine_idx out of bounds")

        return self._tasks[task_idx]["ram_required"] <= self._machines[machine_idx]["ram_limit"]

    @staticmethod
    def gen_items_from_list(num_values, items_list):
        items = np.empty(num_values, dtype=object)
        random_indices = np.random.randint(0, len(items_list), size=num_values)

        for i in range(num_values):
            items[i] = items_list[random_indices[i]]

        return items

    @staticmethod
    def gen_items_from_domain(num_values, variables_domain, same_scale_pos=False):
        items = np.empty(num_values, dtype=object)
        values = {}
        rand_pos = None

        if same_scale_pos:
            rand_pos = np.random.rand(num_values)

        for k, var_range in variables_domain.items():
            if isinstance(var_range[0], int):
                if same_scale_pos:
                    values[k] = np.round(rand_pos * (var_range[1] - var_range[0]) + var_range[0]).astype(int)
                else:
                    values[k] = np.random.randint(var_range[0], var_range[1] + 1, size=num_values)
            elif isinstance(var_range[0], float):
                if same_scale_pos:
                    values[k] = rand_pos * (var_range[1] - var_range[0]) + var_range[0]
                else:
                    values[k] = np.random.rand(num_values) * (var_range[1] - var_range[0]) + var_range[0]

            else:
                raise Exception("invalid type (should be int or float)")
        for i in range(num_values):
            items[i] = {k: values[k][i] for k in values.keys()}
        return items

    def get_tasks(self):
        return self._tasks

    def get_machines(self):
        return self._machines

    def create_pop(self, pop_size, auto_repair=True):
        pop = np.empty((pop_size, len(self._tasks)), dtype=int)
        for i in range(pop_size):
            pop[i] = np.random.randint(0, len(self._machines), size=len(self._tasks))
            if auto_repair:
                pop[i] = self.repair_individual(pop[i])
        return pop

    def get_num_tasks(self):
        return len(self._tasks)

    def get_num_machines(self):
        return len(self._machines)

    def get_cost(self, task_idx, machine_idx):
        return self._cost_matrix[task_idx, machine_idx]

    def get_run_time(self, task_idx, machine_idx):
        return self._run_time_matrix[task_idx, machine_idx]

    def evaluate_individual(self, individual):
        run_time = 0
        cost = 0
        for task_idx, machine_idx in enumerate(individual):
            run_time += self.get_run_time(task_idx, machine_idx)
            cost += self.get_cost(task_idx, machine_idx)

        return np.array([run_time, cost])

    def evaluate_population(self, individuals):
        results = np.zeros((len(individuals), 2))

        for i, individual in enumerate(individuals):
            results[i] = self.evaluate_individual(individual)

        return results

    def check_individual_constraints(self, individual):
        for task_idx, machine_idx in enumerate(individual):
            if not self._is_ram_enough(task_idx, machine_idx):
                return False
        return True

    def get_random_valid_machine(self, task_idx):
        # First list all valid machines
        valid_machines = []
        for machine_idx, machine in enumerate(self.get_machines()):
            if self._is_ram_enough(task_idx, machine_idx):
                valid_machines.append(machine_idx)

        if len(valid_machines) == 0:
            raise Exception("No valid machine found, task can't be run on this set of machines")

        return np.random.choice(valid_machines)  # Then select randomly one of them

    def repair_individual_soft(self, previous_individual, new_individual):
        new_individual = new_individual.astype(int)
        half_vector = (new_individual - previous_individual) / 2

        for task_idx, machine_idx in enumerate(new_individual):
            if new_individual[task_idx] < 0 or new_individual[task_idx] >= len(self.get_machines()):
                corrected_pos = int(previous_individual[task_idx] + half_vector[task_idx])
                if corrected_pos < 0 or corrected_pos >= len(self.get_machines()):
                    corrected_pos = previous_individual[task_idx] + (1 if half_vector[task_idx] > 0 else -1)

                if (
                    corrected_pos < 0
                    or corrected_pos >= len(self.get_machines())
                    or not self._is_ram_enough(task_idx, corrected_pos)
                ):
                    corrected_pos = previous_individual[task_idx]

                new_individual[task_idx] = corrected_pos

        return new_individual

    def repair_individual(self, individual):
        individual = individual.astype(int)
        for task_idx in range(len(individual)):
            # if out of bounds
            if individual[task_idx] < 0:
                individual[task_idx] = 0
            elif individual[task_idx] >= len(self.get_machines()):
                individual[task_idx] = len(self.get_machines()) - 1
            if not self._is_ram_enough(task_idx, individual[task_idx]):
                individual[task_idx] = self.get_random_valid_machine(task_idx)
        return individual

    @staticmethod
    def get_num_objectives():
        return 2

    @staticmethod
    def get_objective_names():
        return ["Run Time (s)", "Cost ($)"]


# TEST

if __name__ == "__main__":
    dataset = Dataset(num_cloud_machines=3, num_fog_machines=10, num_tasks=50, seed=2025)

    pop = dataset.create_pop(pop_size=3)
    print(pop)

    print("task 0:", dataset.get_tasks()[0])
    print("machine 0:", dataset.get_machines()[0])

    print(f"Cost of task 0 on machine 0: {dataset.get_cost(0, 0)}")
    print(f"Time of task 0 on machine 0: {dataset.get_run_time(0, 0)}")
