"""Module that runs MOFDA algorithm, all variables are defined here."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from src.task_offloading_moo.pymoo.algorithms.mofda_optimizer import MOFDAOptimizer  # noqa: E402
from src.task_offloading_moo.pymoo.operators.repair import TaskOffloadingRepair  # noqa: E402
from src.task_offloading_moo.pymoo.operators.sampling import TaskOffloadingSampling  # noqa: E402
from src.task_offloading_moo.pymoo.problem import TaskOffloadingProblem  # noqa: E402
from task_offloading_moo.pymoo.termination.mofda_termination import MOFDATermination  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from pymoo.visualization.scatter import Scatter  # noqa: E402


pop_size = 100
n_max_iters = 50

num_cloud_machines = 30
num_fog_machines = 20
num_tasks = 500

algorithm = MOFDAOptimizer(
    repair=TaskOffloadingRepair(),
    use_soft_repair=True,
    pop_size=pop_size,
    sampling=TaskOffloadingSampling(),
    n_max_iters=n_max_iters,
    archive_size=100,
    save_history=True,
    w=0.4,
    c1=2,
    c2=2,
    beta=4,
    delta=0.5,
)

problem = TaskOffloadingProblem(num_cloud_machines, num_fog_machines, num_tasks)
res = minimize(problem, algorithm, termination=MOFDATermination(n_max_gen=n_max_iters), seed=1, verbose=True)
plot = Scatter(title="MOFDA")
plot.add(res.F)
plot.axis_labels = problem.dataset_generator.get_objective_names()
_ = plot.show()

# Profile
"""
import cProfile
import pstats

def run_minimize(problem, algorithm, n_max_iters):
    print("minimize")
    return minimize(
        problem,
        algorithm,
        termination=MOFDATermination(n_max_gen=n_max_iters),
        seed=1,
        verbose=True
    )
print("profiler")
profiler = cProfile.Profile()
print("run")
profiler.runcall(run_minimize, problem, algorithm, n_max_iters)
print("stats")
stats = pstats.Stats(profiler)
stats.sort_stats(pstats.SortKey.TIME).print_stats()"""
