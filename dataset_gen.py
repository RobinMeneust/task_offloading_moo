# based on https://www.kaggle.com/datasets/sachin26240/vehicularfogcomputing
import numpy as np

# MACHINES DEFINITION
cloud_node_variables_domain = {
    "cpu_rate": [3000,5000],  # MIPS
    "cpu_usage_cost": [0.7, 1.0], # €/h
    "ram_usage_cost": [0.02, 0.05],  # €/GB/h
    "bandwidth_usage_cost": [0.05,0.1], # €/GB
    "latency": [200,300], # ms
}

fog_node_variables_domain = {
    "cpu_rate": [500,1500],  # MIPS
    "cpu_usage_cost": [0.1, 0.4],  # €/h
    "ram_usage_cost": [0.01, 0.03],  # €/GB/h
    "bandwidth_usage_cost": [0.01,0.2],  # €/GB
    "latency": [10,50],  # ms
}

# edge_node ?

num_cloud_machines = 3
num_fog_machines = 10

# TASKS DEFINITION
num_tasks = 10

task_variables_domain = {
    "num_instructions": [1, 1],  # 1 billion (1e9) instructions
    "ram_required": [0.05, 0.2],  # GB
    "in_traffic": [0.01, 0.1],  # GB
    "out_traffic": [0.01, 0.1],  # GB
}


# MACHINES GENERATION

def gen_machines(num_machines, variables_domain):
    machines = np.empty(num_machines, dtype=object)
    machines_values = {}
    for k, var_range in variables_domain.items():
        if type(var_range[0]) == int:
            machines_values[k] = np.random.randint(var_range[0], var_range[1]+1, size=num_machines)
        elif type(var_range[0]) == float:
            machines_values[k] = np.random.rand(num_machines) * (var_range[1] - var_range[0]) + var_range[0]
        else:
            raise Exception("invalid type (should be int or float)")
    for i in range(num_machines):
        machines[i] = {k: machines_values[k][i] for k in machines_values.keys()}
    return machines
        

cloud_machines = gen_machines(num_cloud_machines, cloud_node_variables_domain)
fog_machines = gen_machines(num_fog_machines, fog_node_variables_domain)
machines = np.concatenate((cloud_machines, fog_machines))

# TASKS GENERATION


def gen_tasks(num_tasks, variables_domain):
    tasks = np.empty(num_tasks, dtype=object)
    tasks_values = {}
    for k, var_range in variables_domain.items():
        if type(var_range[0]) == int:
            tasks_values[k] = np.random.randint(var_range[0], var_range[1]+1, size=num_tasks)
        elif type(var_range[0]) == float:
            tasks_values[k] = np.random.rand(num_tasks) * (var_range[1] - var_range[0]) + var_range[0]
        else:
            raise Exception("invalid type (should be int or float)")

    for i in range(num_tasks):
        tasks[i] = {k: tasks_values[k][i] for k in tasks_values.keys()}
    return tasks


tasks = gen_tasks(num_tasks, task_variables_domain)

print(machines[0])
print(tasks[0])

## INDIVIDUAL GEN


def init_pop(pop_size, num_machines, num_tasks):
    pop = np.empty((pop_size, num_tasks), dtype=int)
    for i in range(pop_size):
        pop[i] = np.random.randint(0, num_machines, size=num_tasks)
    return pop


pop_size = 3
pop = init_pop(pop_size, num_cloud_machines+num_cloud_machines, num_tasks)

print(pop)







