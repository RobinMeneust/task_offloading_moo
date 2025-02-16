from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.selection.rnd import RandomSelection
from task_offloading_moo.pymoo.problem import TaskOffloadingProblem
from task_offloading_moo.pymoo.operators.repair import TaskOffloadingRepair
from task_offloading_moo.pymoo.operators.sampling import TaskOffloadingSampling
from pymoo.operators.crossover.pntx import PointCrossover, SinglePointCrossover, TwoPointCrossover

from pymoo.optimize import minimize
from pymoo.operators.mutation.pm import PolynomialMutation

from pymoo.visualization.scatter import Scatter

if __name__ == "__main__":
    pop_size = 20
    algorithm = NSGA2(repair=TaskOffloadingRepair(),
                   pop_size=pop_size,
                   eliminate_duplicates=True,
                   sampling=TaskOffloadingSampling(),
                   selection=RandomSelection(),
                   crossover=SinglePointCrossover(prob=1.0, repair=TaskOffloadingRepair()),
                   mutation=PolynomialMutation(prob=1.0, repair=TaskOffloadingRepair()),
                   )


    num_cloud_machines = 3
    num_fog_machines = 10
    num_tasks = 50

    problem = TaskOffloadingProblem(num_cloud_machines, num_fog_machines, num_tasks)

    res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=True)

    print(f"Best population found:\n {res.X} \nwith F:\n{res.F}")

    plot = Scatter(title="NSGA-II")
    plot.add(res.F)
    plot.axis_labels = problem.dataset_generator.get_objective_names()
    plot.show()
