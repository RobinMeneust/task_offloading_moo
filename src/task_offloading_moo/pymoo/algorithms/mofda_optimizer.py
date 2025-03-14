"""Multi-Objective Flow Direction Algorithm (MOFDA) for multi-objective optimization."""

import sys
import copy
import numpy as np  # noqa: E402
from pymoo.core.algorithm import Algorithm  # noqa: E402
from pymoo.core.population import Population  # noqa: E402
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # noqa: E402
from pymoo.core.individual import Individual  # noqa: E402
from task_offloading_moo.pymoo.archive.mofda_archive import MOFDAArchive  # noqa: E402

from task_offloading_moo.pymoo.problem import TaskOffloadingProblem
from task_offloading_moo.pymoo.operators.repair import TaskOffloadingRepair
from task_offloading_moo.pymoo.operators.sampling import TaskOffloadingSampling
from task_offloading_moo.pymoo.termination.mofda_termination import MOFDATermination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from task_offloading_moo.utils.utils import dominates


# For newer pymoo versions use:
"""
try:
    from pymoo.util.archive import Archive
except ImportError:
    # Fallback for older pymoo versions
    try:
        from pymoo.core.archive import Archive
    except ImportError:
        # Create a simple Archive class if not available
        class Archive(Population):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
"""

# Add the project root to the Python path
sys.path.append("../")


class MOFDAOptimizer(Algorithm):
    """Multi-Objective Flow Direction Algorithm (MOFDA) for multi-objective optimization."""

    def __init__(
        self,
        pop_size=100,
        sampling=None,
        repair=None,
        archive_size=100,
        n_max_iters=100,
        w=0.4,  # weight factor for velocity control
        c1=2,  # control parameter for neighborhood radius
        c2=2,  # control parameter for flow movement
        beta=4,  # number of neighbors around each flow
        delta=0.5,  # mutation rate
        use_soft_repair=True,
        # save_history=False,
        **kwargs,
    ):
        """Initialize the MOFDA algorithm with the given parameters."""
        super().__init__(output=MultiObjectiveOutput(), **kwargs)

        self.pop_size = pop_size
        self.sampling = sampling
        self.repair = repair
        self.n_max_iters = n_max_iters
        self.archive_size = archive_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.beta = beta
        self.delta = delta
        self.use_soft_repair = use_soft_repair
        # self.save_history = save_history

        # Create archive without parameters first
        self.custom_archive = MOFDAArchive(size=archive_size)
        # Then set size attribute separately
        self.custom_archive.size = archive_size

    def copy(self, deep=False):
        """Create a copy of the algorithm."""
        return copy.deepcopy(self) if deep else copy.copy(self)

    def _initialize_infill(self):
        """Initialize the first population if it doesn't exist yet."""
        # Create initial population using the sampling method
        if self.sampling is not None:
            self.pop = self.sampling.do(self.problem, self.pop_size)
        else:
            # Use default random sampling as fallback
            X = np.random.random((self.pop_size, self.problem.n_var))
            # Scale to problem bounds
            xl, xu = self.problem.xl, self.problem.xu
            X = xl + X * (xu - xl)
            # Create population
            self.pop = Population.new("X", X)

        # Evaluate the initial population
        self.evaluator.eval(self.problem, self.pop)

        return self.pop

    def _initialize_advance(self, infills=None, **kwargs):
        # Add initial population to archive
        self.custom_archive.update(self.pop)

    def _repair_solution(self, X):
        """Ensure solutions contain valid integer machine assignments."""
        if X is None:
            return X

        # Make a copy to avoid modifying the original
        X_repaired = X.copy()

        # Get problem bounds
        xl = self.problem.xl
        xu = self.problem.xu

        # Apply bounds and convert to integers
        for i in range(len(X_repaired)):
            # First round to nearest integers
            X_repaired[i] = np.round(X_repaired[i])

            # Then ensure values are within bounds
            X_repaired[i] = np.maximum(X_repaired[i], xl)
            X_repaired[i] = np.minimum(X_repaired[i], xu)

            # Convert to integers explicitly
            X_repaired[i] = X_repaired[i].astype(int)

        return X_repaired

    def _infill(self):
        # Create offspring using FDA operators
        n_offsprings = self.pop_size
        offsprings = np.empty(n_offsprings, dtype=object)

        # Get current population
        X = self.pop.get("X")
        n_vars = X.shape[1]

        # Get leader from archive (best non-dominated solution)
        leader = self.custom_archive.get_leader()
        leader_X = leader.get("X") if leader is not None else None

        # For each individual in population
        for i in range(n_offsprings):
            # Current flow position
            flow_X = X[i % len(X)].copy()

            # Generate neighbors for each flow
            neighbors_X = np.empty((self.beta, n_vars))
            for j in range(self.beta):
                # Create neighbor using normal distribution around current flow
                delta = np.random.randn(n_vars)

                # Calculate neighborhood radius
                radius = delta * np.linalg.norm(leader_X - flow_X) if leader_X is not None else delta

                # Create neighbor
                neighbors_X[j] = flow_X + radius

            # Evaluate neighbors
            # Fixed code using proper Population
            # In the _infill method of MOFDAOptimizer:

            # Before evaluation, make sure all machine assignments are proper integers
            for j in range(len(neighbors_X)):
                # Important: First convert to integer type
                neighbors_X[j] = neighbors_X[j].astype(int)

                # Then use the problem's repair mechanism which knows about constraints
                if self.use_soft_repair and hasattr(self.problem, "soft_repair"):
                    neighbors_X[j] = self.problem.soft_repair(flow_X, neighbors_X[j])
                elif hasattr(self.problem, "repair"):
                    neighbors_X[j] = self.problem.repair(neighbors_X[j])
                else:
                    # Basic bounds repair as fallback
                    xl, xu = self.problem.xl, self.problem.xu
                    neighbors_X[j] = np.clip(neighbors_X[j], xl, xu).astype(int)

            num_machines = self.problem.dataset_generator.get_num_machines()

            for j in range(len(neighbors_X)):
                # Round the solution and cast to integer
                repaired = np.round(neighbors_X[j]).astype(int)
                # Clip values to the valid range
                repaired = np.clip(repaired, 0, num_machines - 1)
                # (Optionally, apply your domain-specific repair if available)
                if self.use_soft_repair and hasattr(self.problem, "soft_repair"):
                    repaired = self.problem.soft_repair(flow_X, repaired)
                neighbors_X[j] = repaired
            neighbors_X = neighbors_X.astype(int)

            # Create population with repaired solutions
            neighbors = Population.new("X", neighbors_X)
            self.evaluator.eval(self.problem, neighbors)

            # Find best neighbor (using non-dominated sorting if multi-objective)
            F_neighbors = np.array([n.get("F") for n in neighbors])
            F_current = self.pop[i].get("F")

            # Check if any neighbor dominates current flow
            dominated = False
            best_neighbor_idx = 0

            for j in range(self.beta):
                if dominates(F_neighbors[j], F_current):
                    dominated = True
                    best_neighbor_idx = j
                    break

            # Update flow position
            if dominated:
                # Calculate flow velocity based on slope
                slope = (flow_X - neighbors_X[best_neighbor_idx]) / np.linalg.norm(
                    flow_X - neighbors_X[best_neighbor_idx]
                )
                velocity = np.random.random() * slope

                # Update flow position
                new_X = flow_X + velocity
            else:
                # Sink filling process - randomly select another flow
                random_idx = np.random.randint(self.pop_size)

                if random_idx == i:
                    random_idx = (random_idx + 1) % self.pop_size

                random_flow = self.pop[random_idx]

                # Compare fitness
                if dominates(random_flow.get("F"), F_current):
                    # Move towards the random flow
                    new_X = flow_X + np.random.random() * (random_flow.get("X") - flow_X)
                elif leader_X is not None:
                    # Move towards the leader
                    new_X = flow_X + 2 * np.random.random() * (leader_X - flow_X)
                else:
                    # Apply mutation
                    if np.random.random() < self.delta:
                        new_X = flow_X + np.random.randn(n_vars) * 0.1
                    else:
                        new_X = flow_X

            # Create new individual
            offsprings[i] = Individual(X=new_X)

        # Apply repair operator if necessary
        if self.repair is not None:
            offsprings = self.repair.do(self.problem, Population(offsprings), use_soft=self.use_soft_repair)

        return offsprings

    def _advance(self, infills=None, **kwargs):
        """Do what is necessary after the infill."""
        # Update archive with evaluated offspring
        self.custom_archive.update(infills)

        # Select next population from combined previous population and offspring
        self.pop = self._select(self.pop, infills)

        # Save algorithm state in history
        # if self.save_history and self.current_iter % 1 == 0:
        #     self.history.append(self.copy(deep=False))

        # self.current_iter += 1

    def _select(self, pop, off):
        # Combine parent and offspring population
        merged = Population.merge(pop, off)

        # For selection, use non-dominated sorting
        F = merged.get("F")

        # Perform non-dominated sorting
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=len(merged))

        # Prepare indices to be selected
        n_remaining = self.pop_size
        indices = []

        for front in fronts:
            # If we can take the whole front, add it
            if len(front) <= n_remaining:
                indices.extend(front)
                n_remaining -= len(front)
            # Otherwise, select based on crowding distance
            else:
                # Calculate crowding distance for this front
                crowding_of_front = self._calc_crowding_distance(F[front])

                # Sort by crowding distance (higher is better)
                crowding_distance = np.argsort(-crowding_of_front)
                crowding_distance = crowding_distance[:n_remaining]

                # Add selected individuals from this front
                indices.extend(front[crowding_distance])
                break

        # Create the population for next generation
        return merged[indices]

    def _calc_crowding_distance(self, F):
        n_points, n_obj = F.shape

        if n_points <= 2:
            return np.full(n_points, np.inf)

        # Initialize crowding distance
        crowding = np.zeros(n_points)

        # Calculate crowding distance for each objective
        for i in range(n_obj):
            # Sort by current objective
            idx = np.argsort(F[:, i])

            # Set the boundary points to infinity
            crowding[idx[0]] = np.inf
            crowding[idx[-1]] = np.inf

            # Calculate crowding distance for the rest
            f_min = F[idx[0], i]
            f_max = F[idx[-1], i]

            if f_max > f_min:
                # Add normalized distance to crowding
                crowding[idx[1:-1]] += (F[idx[2:], i] - F[idx[:-2], i]) / (f_max - f_min)

        return crowding


if __name__ == "__main__":
    # Create the optimizer
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

    # Run the optimizer
    res = minimize(problem, algorithm, seed=1, verbose=True, termination=MOFDATermination(n_max_gen=n_max_iters))

    # Create a scatter plot of the results
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
