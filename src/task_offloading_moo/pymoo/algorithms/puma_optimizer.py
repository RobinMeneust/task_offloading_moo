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
        best_four_pumas_scores_history_explor (np.ndarray): Best four pumas scores history in exploration phase.
        num_unselected_iters_between_best_four_explor (np.ndarray): Number of unselected iterations
            between best four in exploration phase.
        best_four_pumas_scores_history_exploit (np.ndarray): Best four pumas scores history in exploitation phase.
        num_unselected_iters_between_best_four_exploit (np.ndarray): Number of unselected iterations
            between best four in exploitation phase.
        f3_explor (float): F3 score in exploration phase.
        f3_exploit (float): F3 score in exploitation phase.
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

        self.best_four_pumas_scores_history_explor = np.empty((4, self.num_objectives), dtype=float)
        self.num_unselected_iters_between_best_four_explor = np.empty(3, dtype=int)
        self.best_four_pumas_scores_history_exploit = np.empty((4, self.num_objectives), dtype=float)
        self.num_unselected_iters_between_best_four_exploit = np.empty(3, dtype=int)

        self.f3_explor = 0
        self.f3_exploit = 0

        self.alpha_explor = 0.99
        self.alpha_exploit = 0.99

        self.lc = None

        self.archive_size = archive_size
        self._use_archive = use_archive

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
        self.male_puma = RankAndCrowding().do(self.problem, infills, n_survive=1)[0]
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
        # if self._use_archive:
        #     self._update_archive(next_pop)

        return next_pop

    def _advance(self, infills=None, **kwargs):
        """Do what is necessary after the infill."""
        self.pop = infills

    def _finalize(self):
        """Do what is necessary after the optimization (end of the algorithm)."""
        pass

    def unexperienced_phase(self):
        """Run the unexperienced phase to generate the next population.

        Returns:
            Population: Next population.
        """
        current_pop = self.pop
        new_pop = [copy.deepcopy(current_pop)]

        initial_best = RankAndCrowding().do(self.problem, current_pop, n_survive=1)[0].data["rank"]

        for i in range(3):
            self.num_unselected_iters_between_best_four_explor[i] = 1
            self.num_unselected_iters_between_best_four_exploit[i] = 1

        self.best_four_pumas_scores_history_explor[0] = initial_best
        self.best_four_pumas_scores_history_exploit[0] = initial_best

        # apply both exploration and exploitation for 3 iterations
        for i in range(1, 4):
            explor_pop = self.run_exploration(current_pop)
            exploit_pop = self.run_exploitation(current_pop)

            # print("\n\n############\n\n")
            #
            # for x in explor_pop:
            #     print(x.X)
            #
            # print("\n####\n")
            #
            # for x in exploit_pop:
            #     print(x.X)

            # raise Exception("Stop here")

            new_pop.append(explor_pop)
            new_pop.append(exploit_pop)

            male_puma_explor = RankAndCrowding().do(self.problem, explor_pop, n_survive=1)[0]
            male_puma_exploit = RankAndCrowding().do(self.problem, exploit_pop, n_survive=1)[0]

            # update the best solution
            if PumaOptimizer._dominates(male_puma_explor.F, self.male_puma.F):
                self.male_puma = male_puma_explor
            if PumaOptimizer._dominates(male_puma_exploit.F, self.male_puma.F):
                self.male_puma = male_puma_exploit

            # update history (for scores computation)
            self.best_four_pumas_scores_history_explor[i] = male_puma_explor.F
            self.best_four_pumas_scores_history_exploit[i] = male_puma_exploit.F

        new_pop = Population.merge(*new_pop)

        # select best N solutions
        new_pop = RankAndCrowding().do(self.problem, new_pop, n_survive=self.pop_size)

        # update exploration score and exploitation score
        self.update_scores_unexperienced()
        return new_pop

    def experience_phase(self):
        """Run the experienced phase to generate the next population.

        Returns:
            Population: Next population.
        """
        is_explor = False
        new_pop = None

        if self.exploration_score > self.exploitation_score:
            is_explor = True
            new_pop = self.run_exploration(self.pop)
            male_puma = RankAndCrowding().do(self.problem, new_pop, n_survive=1)[0]

            if PumaOptimizer._dominates(male_puma.F, self.male_puma.F):
                # TODO: Use domination ?
                self.male_puma = male_puma

            # update history
            self.num_unselected_iters_between_best_four_explor = np.roll(
                self.num_unselected_iters_between_best_four_explor, -1
            )
            self.num_unselected_iters_between_best_four_explor[-1] = 1

            self.num_unselected_iters_between_best_four_exploit[-1] += 1

            self.best_four_pumas_scores_history_explor = np.roll(self.best_four_pumas_scores_history_explor, -1, axis=0)
            self.best_four_pumas_scores_history_explor[-1] = self.male_puma.F
        else:
            new_pop = self.run_exploitation(self.pop)
            male_puma = RankAndCrowding().do(self.problem, new_pop, n_survive=1)[0]

            if PumaOptimizer._dominates(male_puma.F, self.male_puma.F):
                # TODO: Use domination ?
                self.male_puma = male_puma

            # update history
            self.num_unselected_iters_between_best_four_exploit = np.roll(
                self.num_unselected_iters_between_best_four_exploit, -1
            )
            self.num_unselected_iters_between_best_four_exploit[-1] = 1

            self.num_unselected_iters_between_best_four_explor[-1] += 1

            self.best_four_pumas_scores_history_exploit = np.roll(
                self.best_four_pumas_scores_history_exploit, -1, axis=0
            )
            self.best_four_pumas_scores_history_exploit[-1] = self.male_puma.F

        # update exploration score and exploitation score
        self.update_scores_experienced(is_explor)

        return new_pop

    def update_scores_unexperienced(self):
        """Update the exploration and exploitation scores in the unexperienced phase."""
        # TODO: We can try to use either the rank or the objectives as they are for the cost
        seq_cost_explor = np.empty((3, self.num_objectives))
        seq_cost_exploit = np.empty((3, self.num_objectives))

        pf1_squared = self.pf1**2
        pf2_squared = self.pf2**2

        for i in range(len(seq_cost_explor)):
            seq_cost_explor[i] = np.linalg.norm(
                self.best_four_pumas_scores_history_explor[i] - self.best_four_pumas_scores_history_explor[i + 1], ord=1
            )
            seq_cost_exploit[i] = np.linalg.norm(
                self.best_four_pumas_scores_history_exploit[i] - self.best_four_pumas_scores_history_exploit[i + 1],
                ord=1,
            )

            # TODO: maybe try to use ranks instead of objectives norm
            seq_cost_explor_norm = np.linalg.norm(seq_cost_explor[i], ord=1)
            seq_cost_exploit_norm = np.linalg.norm(seq_cost_exploit[i], ord=1)

            if seq_cost_explor_norm != 0 and (self.lc is None or seq_cost_explor_norm < self.lc):
                self.lc = seq_cost_explor_norm
            if seq_cost_explor_norm != 0 and (self.lc is None or seq_cost_exploit_norm < self.lc):
                self.lc = seq_cost_exploit_norm

        f1_explor = np.linalg.norm(seq_cost_explor[0], ord=1)
        f1_exploit = np.linalg.norm(seq_cost_exploit[0], ord=1)

        f2_explor = np.sum(seq_cost_explor)
        f2_exploit = np.sum(seq_cost_exploit)

        self.exploration_score = pf1_squared * f1_explor + pf2_squared * f2_explor
        self.exploitation_score = pf1_squared * f1_exploit + pf2_squared * f2_exploit

    def update_scores_experienced(self, is_explor):
        """Update the exploration and exploitation scores in the experienced phase.

        Args:
            is_explor (bool): Whether the algorithm is exploring or exploiting.
        """
        # update history
        # TODO: Seq computation for unexperienced and experienced phases should be merged and optimized
        #  (replace best_four_pumas_scores_history_explor)
        seq_cost_explor = np.empty(3)
        seq_cost_exploit = np.empty(3)
        for i in range(len(seq_cost_explor)):
            seq_cost_explor[i] = np.linalg.norm(
                self.best_four_pumas_scores_history_explor[i] - self.best_four_pumas_scores_history_explor[i + 1], ord=1
            )
            seq_cost_exploit[i] = np.linalg.norm(
                self.best_four_pumas_scores_history_exploit[i] - self.best_four_pumas_scores_history_exploit[i + 1],
                ord=1,
            )

        # f1 (escalation)
        f1_explor = seq_cost_explor[-1] / self.num_unselected_iters_between_best_four_explor[-1]
        f1_exploit = seq_cost_exploit[-1] / self.num_unselected_iters_between_best_four_exploit[-1]

        # f2 (resonance)
        f2_explor = np.sum(seq_cost_explor) / np.sum(self.num_unselected_iters_between_best_four_explor)
        f2_exploit = np.sum(seq_cost_exploit) / np.sum(self.num_unselected_iters_between_best_four_exploit)

        # f3 (diversity)
        if is_explor:
            self.f3_explor = 0
            self.f3_exploit += self.pf3
        else:
            self.f3_exploit = 0
            self.f3_explor += self.pf3

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

        # Update lc
        if seq_cost_explor[-1] != 0 and (self.lc is None or seq_cost_explor[-1] < self.lc):
            self.lc = seq_cost_explor[-1]
        if seq_cost_exploit[-1] != 0 and (self.lc is None or seq_cost_exploit[-1] < self.lc):
            self.lc = seq_cost_exploit[-1]

        # TODO: maybe try to use ranks instead of objectives norm
        # seq_cost_explor_norm = np.linalg.norm(seq_cost_explor[-1], ord=1)
        # seq_cost_exploit_norm = np.linalg.norm(seq_cost_exploit[-1], ord=1)

        # compute final scores
        self.exploration_score = self.alpha_explor * (f1_explor + f2_explor) + delta_explor * self.lc * self.f3_explor
        self.exploitation_score = (
            self.alpha_exploit * (f1_exploit + f2_exploit) + delta_exploit * self.lc * self.f3_exploit
        )

    @staticmethod
    def _dominates(x, z):
        """Check if x dominates z.

        TODO: (Suggestion) Move this method to a utility class in another file.

        Args:
            x (np.ndarray): First solution.
            z (np.ndarray): Second solution.

        Returns:
            bool: Whether x dominates z or not.
        """
        no_worse = all(x_i <= z_i for x_i, z_i in zip(x, z))
        strictly_better = any(x_i < z_i for x_i, z_i in zip(x, z))

        return no_worse and strictly_better

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

                a, b, c, d, e, f = np.random.choice(len(new_pop), 6, replace=False)
                vec_ba = new_pop[a].X - new_pop[b].X
                vec_dc = new_pop[c].X - new_pop[d].X
                vec_fe = new_pop[e].X - new_pop[f].X

                zi = xi.copy()

                # TODO: remove vec_dc here, it was just kept for now to have the same formula has the paper
                zi.X = new_pop[a].X + g * (vec_ba + vec_ba - vec_dc + vec_dc - vec_fe)

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
            if PumaOptimizer._dominates(zi.F, xi.F):
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
                if np.random.rand() < self.l_prob:
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
                    f1 = np.random.randn(dim) * np.exp(2 - (self.current_iter / self.n_max_iters))
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

            if PumaOptimizer._dominates(zi.F, xi.F):
                new_pop[i] = zi

        return new_pop

    # def _update_archive(self, pop):
    #     # Merge the archive and current population
    #     merged = Population.merge(self.archive, pop)
    #
    #     num_selected = self.archive_size if self.archive_size is not None else len(merged)
    #     new_archive = RankAndCrowding().do(self.problem, merged, n_survive=num_selected)
    #
    #     # Remove dominated solutions
    #     nds_indices = np.unique(new_archive.get("rank"), return_index=True)[1]
    #     new_archive = new_archive[nds_indices]
    #
    #     self.archive = new_archive
