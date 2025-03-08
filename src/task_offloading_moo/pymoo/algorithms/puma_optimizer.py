"""This file contains the (Pymoo) operators for the task offloading problem."""

from pymoo.util.archive import default_archive  # MultiObjectiveArchive
import copy
import numpy as np
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.algorithm import Algorithm
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.repair import NoRepair
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination import get_termination
from pymoo.util.display.column import Column
from enum import Enum
from task_offloading_moo.utils.utils import dominates


class Mode(int, Enum):
    """Mode of the PUMA algorithm (exploration or exploitation)."""

    EXPLORE = 0
    EXPLOIT = 1


class PumaOutput(MultiObjectiveOutput):
    """Output class for PUMA algorithm.

    Attributes:
        explor (Column): Exploration score column.
        exploit (Column): Exploitation score column.
        is_explore (Column): Whether the algorithm is exploring or exploiting.
    """

    def __init__(self):
        """Initialize the output class."""
        super().__init__()
        self.explor = Column("explor", width=13)
        self.exploit = Column("exploit", width=13)
        self.is_explore = Column("is_explore", width=13)
        self.columns += [self.explor, self.exploit, self.is_explore]

    def update(self, algorithm):
        """Update the output with the given PUMA Optimizer state.

        Args:
            algorithm (PumaOptimizer): Algorithm to update the output with.
        """
        super().update(algorithm)

        self.explor.set(algorithm.exploration_score)
        self.exploit.set(algorithm.exploitation_score)
        self.is_explore.set("yes" if algorithm.exploration_score > algorithm.exploitation_score else "no")


class PumaOptimizer(Algorithm):
    """PUMA optimizer.

    Attributes:
        use_soft_repair (bool): Whether to use soft repair or not.
        current_iter (int): Current iteration.
        n_max_iters (int): Maximum number of iterations.
        initialization (Initialization): Initialization method.
        pop_size (int): Population size.
        repair (Repair): Repair method.
        male_puma (Individual): Best solution.
        is_unexperienced (bool): Whether the algorithm is in the unexperienced phase.
        exploration_score (float): Exploration score.
        exploitation_score (float): Exploitation score.
        pf1 (float): Weight of function 1 (escalation).
        pf2 (float): Weight of function 2 (resonance).
        pf3 (float): Weight of function 3 (diversity).
        u_prob (float): Probability in the exploration phase of changing a component with a new value.
        l_prob (float): Probability that the ambush type is "small jump towards 2 other pumas" instead of
            "long jump towards the best puma".
        alpha (float): The higher it is, the smaller the new puma components will be in the run strategy in
            exploitation phase, and thus the higher is the perturbation.
        num_objectives (int): Number of objectives.
        seq_cost (np.ndarray): Sequence of costs, i.e. distances between the male puma (i.e. best)
            across all runs and the current male puma.
        seq_time (np.ndarray): Sequence of times, i.e. number of iterations between modes changes for each mode.
        modes_num_unselected_iters (np.ndarray): Number of iterations for each mode
            since last selection (i.e. last mode change).
        f3 (np.ndarray): F3 score for each mode.
        prev_mode (Mode): Mode in previous iteration.
        alpha_explor (float): Alpha in exploration phase.
        alpha_exploit (float): Alpha in exploitation phase.
        lc (float): LC value.
        archive_size (int): Size of the archive.
        _use_archive (bool): Whether to use the archive or not.
    """

    def __init__(
        self,
        use_soft_repair=False,
        pf1=0.5,
        pf2=0.5,
        pf3=0.3,
        u=0.2,
        l=0.9,
        alpha=2,
        pop_size=25,
        n_max_iters=20,
        sampling=LHS(),
        repair=NoRepair(),
        output=PumaOutput(),
        num_objectives=2,
        archive_size=25,
        use_archive=True,
        num_archive_injections=0,
        archive_injections_prob=0.2,
        **kwargs
    ):
        """Initialize PUMA optimizer.

        Args:
            use_soft_repair (bool): Whether to use soft repair or not. Defaults to False
            pf1 (float): Weight of function 1 (escalation). Defaults to 0.5.
                It uses distance between the best solution in initial population and at iter 1 (in unexperienced phase),
                or between the best solution in previous population and at current iter (in experienced phase).
            pf2 (float): Weight of function 2 (resonance). Defaults to 0.5. same as function 1,
                but considers 4 populations best solutions and sums distances.
            pf3 (float): Weight of function 3 (diversity). Defaults to 0.3.
                Phases (exploration/exploitation) that hasn't been used for many iterations are favored.
            u (float): Probability in the exploration phase of changing a component with a new value. Defaults to 0.2.
            l (float): Probability that the ambush type is "small jump towards 2 other pumas" (including the best)
                instead of "long jump towards the best puma". Defaults to 0.9.
            alpha (float): The higher it is, the smaller the new puma components will be in the run strategy
                in exploitation phase, and thus the higher is the perturbation. Defaults to 2.
            pop_size (int): Population size. Defaults to 25.
            n_max_iters (int): Maximum number of iterations. Defaults to 20. Minimum value is 3.
            sampling (Sampling): Sampling method. Defaults to LHS.
            repair (Repair): Repair method. Defaults to NoRepair.
            output (Output): Output method. Defaults to PumaOutput.
            num_objectives (int): Number of objectives. Defaults to 2.
            archive_size (int): Size of the archive. If None, then no limit is set. Defaults to 25.
            use_archive (bool): Whether to use the archive or not. Defaults to True.
            num_archive_injections (int): Number of archive injections. Defaults to 0.
            archive_injections_prob (float): Probability of injecting solutions from the archive. Defaults to 0.2.
        """
        if n_max_iters < 3:
            raise ValueError("The number of generations must be at least 3.")

        super().__init__(output=output, termination=get_termination("n_gen", n_max_iters - 3), **kwargs)

        self.use_soft_repair = use_soft_repair

        self.current_iter = 0
        self.n_max_iters = n_max_iters

        self.initialization = Initialization(sampling)

        self.pop_size = pop_size
        self.repair = repair

        self.male_puma = None  # best solution

        self.is_unexperienced = True

        self.exploration_score = 0
        self.exploitation_score = 0

        self.pf1 = pf1
        self.pf2 = pf2
        self.pf3 = pf3
        self.u_prob = u
        self.l_prob = l
        self.alpha = alpha
        self.num_objectives = num_objectives

        self.seq_cost = np.empty((len(Mode), 3), dtype=float)
        self.seq_time = np.empty((len(Mode), 3), dtype=float)
        self.modes_num_unselected_iters = np.empty(len(Mode), dtype=int)

        self.f3 = np.zeros(len(Mode), dtype=float)

        self.alpha_explor = 0.99
        self.alpha_exploit = 0.99

        self.lc = None

        self.archive_size = archive_size
        self._use_archive = use_archive

        self.prev_mode = Mode.EXPLORE

        self.num_archive_injections = num_archive_injections
        self.archive_injections_prob = archive_injections_prob

    def _setup(self, problem, **kwargs):
        """Set up the algorithm.

        Args:
            problem (Problem): Problem to solve.
        """
        if self._use_archive:
            # self.archive = MultiObjectiveArchive(max_size=archive_size, truncate_size=None)
            self.archive = default_archive(self.problem, max_size=self.archive_size)

    def _initialize_infill(self):
        """Initialize the population.

        Returns:
            Population: Initial population.
        """
        init_pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return init_pop

    def _initialize_advance(self, infills=None, **kwargs):
        """Do what is necessary after the initialization."""
        self.male_puma = self.get_male_puma(infills)
        # if self._use_archive:
        #     self._update_archive(init_pop)

    def _infill(self):
        """Create the next population.

        Returns:
            Population: Next population.
        """
        if self.is_unexperienced:
            next_pop = self.unexperienced_phase()
            self.is_unexperienced = False
            self.current_iter += 3
        else:
            next_pop = self.experience_phase()
            self.current_iter += 1

        if self._use_archive and self.num_archive_injections > 0 and np.random.rand() < self.archive_injections_prob:
            num_injections = min(self.num_archive_injections, len(self.archive))
            injections_indices = np.random.choice(range(len(self.archive)), num_injections, replace=False)
            injections = self.archive[injections_indices]
            next_pop = [next_pop, injections]
            next_pop = Population.merge(*next_pop)

        return next_pop

    def _advance(self, infills=None, **kwargs):
        """Do what is necessary after the infill."""
        self.pop = infills

    def _finalize(self):
        """Do what is necessary after the optimization (end of the algorithm)."""
        pass

    def compute_distance_cost(self, ind1, ind2, pop=None):
        """Compute the distance cost between two individuals.

        We consider we compare ind1 relative to ind2. If ind1 is worse then distance is 0.

        Args:
            ind1 (Individual): First individual.
            ind2 (Individual): Second individual.
            pop (Population): Population to consider. Defaults to None (i.e. use self.pop for distance computation).

        Returns:
            float: Distance cost.
        """
        # return np.linalg.norm(ind1.F - ind2.F, ord=1) # L1 norm version

        weight_rank = 0.8

        if pop is None:
            pop = self.pop
        pop = copy.deepcopy(pop)

        # check if ind1.X is in pop.get("X")
        if not np.any(np.all(pop.get("X") == ind1.X, axis=1)):
            new_pop = Population.new("X", np.array([ind1.X]))
            pop = Population.merge(pop, new_pop)
        if not np.any(np.all(pop.get("X") == ind2.X, axis=1)):
            new_pop = Population.new("X", np.array([ind2.X]))
            pop = Population.merge(pop, new_pop)
        pop = RankAndCrowding().do(self.problem, pop)

        # get back the individuals from X values
        ind1 = pop[np.all(pop.get("X") == ind1.X, axis=1)][0]
        ind2 = pop[np.all(pop.get("X") == ind2.X, axis=1)][0]

        rank_diff = np.abs(ind1.data["rank"] - ind2.data["rank"])
        crowding_diff = np.abs(ind1.data["crowding"] - ind2.data["crowding"])

        if np.isnan(crowding_diff):
            crowding_diff = 0

        # clip values and consider the case where ind2 is better than ind1
        if ind1.data["rank"] > ind2.data["rank"]:  # ind1 is worse (low rank is better)
            return 0
        elif rank_diff > 10:
            rank_diff = 10

        if ind1.data["crowding"] < ind2.data["crowding"]:  # same as above (high crowding is better)
            crowding_diff = 0
        elif crowding_diff > 10:
            crowding_diff = 10

        return weight_rank * rank_diff + (1 - weight_rank) * crowding_diff

    def get_male_puma(self, pop):
        """Get the best solution from the population (male puma). Randomly selects from the lowest rank.

        Args:
            pop (Population): Population to get the best solution from.

        Returns:
            Individual: Best solution.
        """
        sorted_pop = copy.deepcopy(pop)

        # Rank and crowding sort
        sorted_pop = RankAndCrowding().do(self.problem, sorted_pop)

        # We keep lowest rank values only
        min_rank = min(ind.data["rank"] for ind in sorted_pop)
        elite_pop = [ind for ind in sorted_pop if ind.data["rank"] == min_rank]

        # Choose a solution randomly from the best solutions
        male_puma = np.random.choice(elite_pop, 1)[0]
        return male_puma

    def unexperienced_phase(self):
        """Run the unexperienced phase to generate the next population.

        Returns:
            Population: Next population.
        """
        initial_best = RankAndCrowding().do(self.problem, self.pop, n_survive=1)[0]
        prev_best_explor = initial_best
        prev_best_exploit = initial_best

        self.seq_time.fill(1)

        # apply both exploration and exploitation for 3 iterations
        for i in range(0, 3):
            new_pop = [copy.deepcopy(self.pop)]

            explor_pop = self.run_exploration(self.pop)
            exploit_pop = self.run_exploitation(self.pop)

            new_pop.append(explor_pop)
            new_pop.append(exploit_pop)

            # Choose a solution randomly from the best solutions
            male_puma_explor = self.get_male_puma(explor_pop)
            male_puma_exploit = self.get_male_puma(exploit_pop)

            # male_puma_explor = RankAndCrowding().do(self.problem, explor_pop, n_survive=1)[0]
            # male_puma_exploit = RankAndCrowding().do(self.problem, exploit_pop, n_survive=1)[0]

            # Update pop (select best N solutions)
            new_pop = Population.merge(*new_pop)
            new_pop = RankAndCrowding().do(self.problem, new_pop, n_survive=self.pop_size)

            # update the best solution
            self.male_puma = self.get_male_puma(new_pop)

            # update history (for scores computation)
            self.update_seq_cost_lc(prev_best_explor, male_puma_explor, Mode.EXPLORE, i)
            self.update_seq_cost_lc(prev_best_exploit, male_puma_exploit, Mode.EXPLOIT, i)

            # Update prev_best_F
            prev_best_explor = male_puma_explor
            prev_best_exploit = male_puma_exploit

            self.pop = new_pop

        # update exploration score and exploitation score
        self.update_scores_unexperienced()
        return new_pop

    def update_seq_time(self):
        """Update the sequence of times for each mode. That is, the number of iterations since the last mode change."""
        self.seq_time = np.roll(self.seq_time, -1, axis=1)  # shift
        self.seq_time[Mode.EXPLORE, -1] = self.modes_num_unselected_iters[Mode.EXPLORE]
        self.seq_time[Mode.EXPLOIT, -1] = self.modes_num_unselected_iters[Mode.EXPLOIT]

    def experience_phase(self):
        """Run the experienced phase to generate the next population.

        Returns:
            Population: Next population.
        """
        new_pop = None

        if self.exploration_score > self.exploitation_score:
            new_pop = self.run_exploration(self.pop)
            mode = Mode.EXPLORE
        else:
            new_pop = self.run_exploitation(self.pop)
            mode = Mode.EXPLOIT

        male_puma = self.get_male_puma(new_pop)

        # update history (for scores computation) (except num_unselected_iters)
        self.update_seq_cost_lc(self.male_puma, male_puma, mode, pop=new_pop)

        # update the best solution
        if dominates(male_puma.F, self.male_puma.F):
            self.male_puma = male_puma

        # update exploration score and exploitation score
        self.update_scores_experienced()

        # update seq_time (if mode change) and modes_num_unselected_iters (must be updated after seq_time)
        if self.prev_mode != mode:
            self.prev_mode = mode
            self.update_seq_time()

        self.modes_num_unselected_iters[mode] = 1
        self.modes_num_unselected_iters[1 - mode] += 1

        return new_pop

    def update_seq_cost_lc(self, prev_best, current_best, mode: Mode, idx=-1, pop=None):
        """Update the seq_cost and lc values for given mode (explor/exploit).

        Args:
            prev_best (Individual): Previous best solution.
            current_best (Individual): Current best solution.
            mode (Mode): Mode of the algorithm (EXPLORE or EXPLOIT).
            idx (int): Index of the sequence of costs. Defaults to -1 (considered full, so we shift values.
            pop (Population): Population to consider. Defaults to None (i.e. use self.pop for distance computation).
        """
        if idx == -1:
            # considered full (so we shift)
            self.seq_cost[mode] = np.roll(self.seq_cost[mode], -1, axis=0)

        self.seq_cost[mode, idx] = self.compute_distance_cost(prev_best, current_best, pop)

        if self.seq_cost[mode, idx] != 0 and (self.lc is None or self.seq_cost[mode, idx] < self.lc):
            self.lc = self.seq_cost[mode, idx]

    def compute_escalation_score(self):
        """Compute the escalation score (f1)."""
        return self.seq_cost[:, 0] / self.seq_time[:, -1]

    def compute_resonance_score(self):
        """Compute the resonance score (f2)."""
        return np.sum(self.seq_cost, axis=1) / np.sum(self.seq_time, axis=1)

    def update_diversity_score(self):
        """Compute the diversity score (f3)."""
        is_explor = self.exploration_score > self.exploitation_score

        if is_explor:
            self.f3[Mode.EXPLORE] = 0
            self.f3[Mode.EXPLOIT] += self.pf3
        else:
            self.f3[Mode.EXPLOIT] = 0
            self.f3[Mode.EXPLORE] += self.pf3

    def update_scores_unexperienced(self):
        """Update the exploration and exploitation scores for the unexperienced phase."""
        # f1 (escalation)
        f1 = self.compute_escalation_score()

        # f2 (resonance)
        f2 = self.compute_resonance_score()

        pf1_squared = self.pf1**2
        pf2_squared = self.pf2**2

        self.exploration_score = pf1_squared * f1[Mode.EXPLORE] + pf2_squared * f2[Mode.EXPLORE]
        self.exploitation_score = pf1_squared * f1[Mode.EXPLOIT] + pf2_squared * f2[Mode.EXPLOIT]

    def update_scores_experienced(self):
        """Update the exploration and exploitation scores for the experienced phase."""
        # f1 (escalation)
        f1 = self.compute_escalation_score()

        # f2 (resonance)
        f2 = self.compute_resonance_score()

        # f3 (diversity)
        self.update_diversity_score()

        # Update alphas
        if self.exploration_score < self.exploitation_score:
            self.alpha_explor = max((self.alpha_explor - 0.01), 0.01)
            self.alpha_exploit = 0.99
        else:
            self.alpha_explor = 0.99
            self.alpha_exploit = max((self.alpha_exploit - 0.01), 0.01)

        # Compute deltas
        delta_explor = 1 - self.alpha_explor
        delta_exploit = 1 - self.alpha_exploit

        # compute final scores
        self.exploration_score = (
            self.alpha_explor * (f1[Mode.EXPLORE] + f2[Mode.EXPLORE]) + delta_explor * self.lc * self.f3[Mode.EXPLORE]
        )
        self.exploitation_score = (
            self.alpha_exploit * (f1[Mode.EXPLOIT] + f2[Mode.EXPLOIT]) + delta_exploit * self.lc * self.f3[Mode.EXPLOIT]
        )

    def run_exploration(self, current_pop):
        """Run the exploration phase to generate the next population.

        Args:
            current_pop (Population): Current population.

        Returns:
            Population: Next population.
        """
        u = self.u_prob
        p = (1 - u) / (len(current_pop))
        new_pop = copy.deepcopy(current_pop)

        # sort the population in ascending order of rank (i.e. the best solutions first)
        new_pop = RankAndCrowding().do(self.problem, new_pop, n_survive=len(new_pop))

        for i in range(len(new_pop)):
            xi = new_pop[i]

            if np.random.rand() < 0.5:
                # create a completely random solution zi
                # repair is normally already included in the sampling
                zi = self.initialization.do(self.problem, 1, algorithm=self)[0]
            else:
                # select 6 random distinct solutions (different from xi) to build zi
                # create a solution between pumas
                g = np.random.rand() * 2 - 1  # in [-1,1]

                # a, b, c, d, e, f = np.random.choice(len(new_pop), 6, replace=False)
                a, b, e, f = np.random.choice(len(new_pop), 4, replace=False)
                vec_ba = new_pop[a].X - new_pop[b].X
                # vec_dc = new_pop[c].X - new_pop[d].X
                vec_fe = new_pop[e].X - new_pop[f].X

                zi = xi.copy()

                # paper's formula is the following one (commented), but vec_dc cancels out, so we simplify it
                # zi.X = new_pop[a].X + g * (vec_ba + vec_ba - vec_dc + vec_dc - vec_fe)

                zi.X = new_pop[a].X + g * (2 * vec_ba - vec_fe)

                # repair the solution
                if self.use_soft_repair:
                    zi.X = self.problem.soft_repair(xi.X, zi.X)
                else:
                    zi.X = self.repair(self.problem, Population.new("X", [zi.X]))[0].X

            # kind of crossover between xi and zi
            j0 = np.random.randint(0, len(xi.X))  # xj0 will become zj0

            for j in range(len(xi.X)):
                if j != j0 or np.random.rand() >= u:
                    # keep initial value
                    zi.X[j] = xi.X[j]

            # evaluate the new solution
            zi.F = self.problem.evaluate([zi.X])[0]

            # update u probability
            if dominates(zi.F, xi.F):
                new_pop[i] = zi
            else:
                u += p

        return new_pop

    def run_exploitation(self, current_pop):
        """Run the exploitation phase to generate the next population.

        Args:
            current_pop (Population): Current population.

        Returns:
            Population: Next population.
        """
        dim = len(current_pop[0].X)
        new_pop = copy.deepcopy(current_pop)

        for i in range(len(new_pop)):
            xi = new_pop[i]
            if np.random.rand() < 0.5:
                # Ambush strategy
                if np.random.rand() > self.l_prob:
                    # Small jump towards 2 other pumas
                    random_puma_x = self.initialization.do(self.problem, 1, algorithm=self)[0].X
                    random_puma_scaled_x = random_puma_x * 2 * np.random.rand() * np.exp(np.random.randn(dim))

                    zi = xi.copy()
                    zi.X = self.male_puma.X + random_puma_scaled_x - xi.X
                else:
                    # Long jump towards the best puma
                    factor = 2 * np.random.rand()
                    denominator = 2 * np.random.rand() - 1 + np.random.randn(dim)

                    r = 2 * np.random.rand() - 1
                    f1 = np.random.randn(dim) * np.exp(2 - 2 * (self.current_iter / self.n_max_iters))
                    w = np.random.randn(dim)
                    v = np.random.randn(dim)
                    f2 = w * (v**2) * np.cos(2 * np.random.rand() * w)

                    scaled_xi = f1 * r * xi.X
                    scaled_puma_male = f2 * (1 - r) * self.male_puma.X

                    zi = xi.copy()
                    zi.X = (factor * (scaled_xi + scaled_puma_male) / denominator) - self.male_puma.X
            else:
                # Run strategy
                mean_puma = np.mean(np.array([ind.X for ind in new_pop]), axis=0) / len(new_pop)
                random_puma = self.initialization.do(self.problem, 1, algorithm=self)[0]
                beta = np.random.randint(0, 2)  # 0 or 1
                denominator = 1 + np.random.rand() * self.alpha  # scale

                zi = xi.copy()
                zi.X = (np.multiply(random_puma.X, mean_puma) - (1**beta) * xi.X) / denominator  # element wise product

            # repair the solution
            if self.use_soft_repair:
                zi.X = self.problem.soft_repair(xi.X, zi.X)
            else:
                zi.X = self.repair(self.problem, Population.new("X", [zi.X]))[0].X

            # evaluate the new solutions
            zi.F = self.problem.evaluate([zi.X])[0]

            if dominates(zi.F, xi.F):
                new_pop[i] = zi

        return new_pop
