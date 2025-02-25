import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.algorithm import Algorithm
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.repair import NoRepair
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination import get_termination


class PumaOutput(MultiObjectiveOutput):

    def __init__(self):
        super().__init__()

        # TODO: If no fields are added here, remove this class and use MultiObjectiveOutput directly

    def update(self, algorithm):
        super().update(algorithm)

        # TODO: Add instructions here or remove this method


class PumaOptimizer(Algorithm):
    def __init__(
        self,
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
        **kwargs
    ):
        """Initializes PUMA optimizer.

        Args:
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
        """

        if n_max_iters < 3:
            raise ValueError("The number of generations must be at least 3.")

        super().__init__(output=output, termination=get_termination("n_gen", n_max_iters - 3), **kwargs)

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

        self.best_four_pumas_scores_history_explor = np.empty(4, dtype=float)
        self.num_unselected_iters_between_best_four_explor = np.empty(3, dtype=int)
        self.best_four_pumas_scores_history_exploit = np.empty(4, dtype=float)
        self.num_unselected_iters_between_best_four_exploit = np.empty(3, dtype=int)

        self.f3_explor = 0
        self.f3_exploit = 0

        self.alpha_explor = 0.99
        self.alpha_exploit = 0.99

        self.lc = None

    def _setup(self, problem, **kwargs):
        pass

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        pass

    def _infill(self):
        pass
        # if self.is_unexperienced:
        #     self.unexperienced_phase()
        #     self.is_unexperienced = False
        #     self.current_iter += 3
        # else:
        #     self.experience_phase()
        #     self.current_iter += 1
        #
        #
        #
        # self.current_iter
        #
        # problem, particles, pbest = self.problem, self.particles, self.pop
        #
        # (X, V) = particles.get("X", "V")
        # P_X = pbest.get("X")
        #
        # sbest = self._social_best()
        # S_X = sbest.get("X")
        #
        # Xp, Vp = pso_equation(X, P_X, S_X, V, self.V_max, self.w, self.c1, self.c2)
        #
        # # create the offspring population
        # off = Population.new(X=Xp, V=Vp)
        #
        # # try to improve the current best with a pertubation
        # if self.pertube_best:
        #     k = FitnessSurvival().do(problem, pbest, n_survive=1, return_indices=True)[0]
        #     mut = PM(prob=0.9, eta=np.random.uniform(5, 30), at_least_once=False)
        #     mutant = mut(problem, Population(Individual(X=pbest[k].X)))[0]
        #     off[k].set("X", mutant.X)
        #
        # self.repair(problem, off)
        # self.sbest = sbest
        #
        # return off

    def _advance(self, infills=None, **kwargs):
        pass

    def _finalize(self):
        pass

    def unexperienced_phase(self):
        current_pop = self.pop
        new_pop = [current_pop]

        initial_best = FitnessSurvival().do(self.problem, current_pop, n_survive=1)[0].F

        for i in range(3):
            self.num_unselected_iters_between_best_four_explor[i] = 1
            self.num_unselected_iters_between_best_four_exploit[i] = 1

        self.best_four_pumas_scores_history_explor[0] = initial_best
        self.best_four_pumas_scores_history_exploit[0] = initial_best

        # apply both exploration and exploitation for 3 iterations
        for i in range(1, 4):
            explor_pop = self.run_exploration(current_pop)
            exploit_pop = self.run_exploitation(current_pop)

            new_pop.append(explor_pop)
            new_pop.append(exploit_pop)

            male_puma_explor = FitnessSurvival().do(self.problem, explor_pop, n_survive=1)[0]
            male_puma_exploit = FitnessSurvival().do(self.problem, exploit_pop, n_survive=1)[0]

            # update the best solution
            if male_puma_explor.F < self.male_puma.F:
                self.male_puma = male_puma_explor
            if male_puma_exploit.F < self.male_puma.F:
                self.male_puma = male_puma_exploit

            # update history (for scores computation)
            self.best_four_pumas_scores_history_explor[i] = male_puma_explor.F
            self.best_four_pumas_scores_history_explor[i] = male_puma_exploit.F

        self.pop = Population.merge(*new_pop)

        # select best N solutions
        self.pop = FitnessSurvival().do(self.problem, self.pop, n_survive=self.pop_size)

        # update exploration score and exploitation score
        self.update_scores_unexperienced()

    def experience_phase(self):
        is_explor = False
        if self.exploration_score > self.exploitation_score:
            is_explor = True
            self.run_exploration(self.pop)

            # update history
            self.num_unselected_iters_between_best_four_explor = np.roll(
                self.num_unselected_iters_between_best_four_explor, -1
            )
            self.num_unselected_iters_between_best_four_explor[-1] = 1

            self.num_unselected_iters_between_best_four_exploit[-1] += 1
        else:
            self.run_exploitation(self.pop)

            # update history
            self.num_unselected_iters_between_best_four_exploit = np.roll(
                self.num_unselected_iters_between_best_four_exploit, -1
            )
            self.num_unselected_iters_between_best_four_exploit[-1] = 1

            self.num_unselected_iters_between_best_four_explor[-1] += 1

        male_puma = FitnessSurvival().do(self.problem, self.pop, n_survive=1)[0]

        # TODO: Note that this section is unclear in the paper,
        #  and seems to be different in their code, so we should check it again
        if male_puma.F < self.male_puma.F:
            # update the best solution
            self.male_puma = male_puma

            # update history
            if is_explor:
                self.best_four_pumas_scores_history_explor = np.roll(self.best_four_pumas_scores_history_explor, -1)
                self.best_four_pumas_scores_history_explor[-1] = male_puma.F
            else:
                self.best_four_pumas_scores_history_exploit = np.roll(self.best_four_pumas_scores_history_exploit, -1)
                self.best_four_pumas_scores_history_exploit[-1] = male_puma.F
        else:
            self.num_unselected_iters_between_best_four_explor[-1] += 1
            self.num_unselected_iters_between_best_four_exploit[-1] += 1

        # update exploration score and exploitation score
        self.update_scores_experienced(is_explor)

    def update_scores_unexperienced(self):
        # TODO: We can try to use either the rank or the objectives as they are for the cost

        seq_cost_explor = np.empty(3)
        seq_cost_exploit = np.empty(3)

        pf1_squared = self.pf1**2
        pf2_squared = self.pf2**2

        for i in range(len(seq_cost_explor)):
            seq_cost_explor[i] = abs(
                self.best_four_pumas_scores_history_explor[i] - self.best_four_pumas_scores_history_explor[i + 1]
            )
            seq_cost_exploit[i] = abs(
                self.best_four_pumas_scores_history_exploit[i] - self.best_four_pumas_scores_history_exploit[i + 1]
            )

            if seq_cost_explor[i] != 0 and (self.lc is None or seq_cost_explor[i] < self.lc):
                self.lc = seq_cost_explor[i]
            if seq_cost_exploit[i] != 0 and (self.lc is None or seq_cost_exploit[i] < self.lc):
                self.lc = seq_cost_exploit[i]

        f1_explor = seq_cost_explor[0]
        f1_exploit = seq_cost_exploit[0]

        f2_explor = np.sum(seq_cost_explor)
        f2_exploit = np.sum(seq_cost_exploit)

        self.exploration_score = pf1_squared * f1_explor + pf2_squared * f2_explor
        self.exploitation_score = pf1_squared * f1_exploit + pf2_squared * f2_exploit

    def update_scores_experienced(self, is_explor):
        # update history
        # TODO: Seq computation for unexperienced and experienced phases should be merged and optimized
        #  (replace best_four_pumas_scores_history_explor)
        seq_cost_explor = np.empty(3)
        seq_cost_exploit = np.empty(3)
        for i in range(len(seq_cost_explor)):
            seq_cost_explor[i] = abs(
                self.best_four_pumas_scores_history_explor[i] - self.best_four_pumas_scores_history_explor[i + 1]
            )
            seq_cost_exploit[i] = abs(
                self.best_four_pumas_scores_history_exploit[i] - self.best_four_pumas_scores_history_exploit[i + 1]
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

        # compute final scores
        self.exploration_score = self.alpha_explor * (f1_explor + f2_explor) + delta_explor * self.lc * self.f3_explor
        self.exploitation_score = (
            self.alpha_exploit * (f1_exploit + f2_exploit) + delta_exploit * self.lc * self.f3_exploit
        )

    def run_exploration(self, current_pop):
        p = (1 - self.u_prob) / (len(current_pop))

        for i in range(len(current_pop)):
            xi = current_pop[i]

            if np.random.rand() < 0.5:
                # create a completely random solution zi
                # repair is normally already included in the sampling
                zi = self.initialization.do(self.problem, 1, algorithm=self)[0]
            else:
                # select 6 random distinct solutions (different from xi) to build zi
                # create a solution between pumas
                g = np.random.rand() * 2 - 1  # in [-1,1]

                xa, xb, xc, xd, xe, xf = np.random.choice(len(current_pop), 6, replace=False)
                vec_ba = current_pop[xa].X - current_pop[xb].X
                vec_dc = current_pop[xc].X - current_pop[xd].X
                vec_fe = current_pop[xe].X - current_pop[xf].X

                zi = xi.copy()

                # TODO: remove vec_dc here, it was just kept for now to have the same formula has the paper
                zi.X = xa.X + g * (vec_ba + vec_ba - vec_dc + vec_dc - vec_fe)

                # repair the solution
                zi = self.repair(self.problem, zi)

            # kind of crossover between xi and zi
            j0 = np.random.randint(0, len(xi.X))  # xj0 will become zj0

            for j in range(len(xi.X)):
                if j != j0 or np.random.rand() >= self.u_prob:
                    # keep initial value
                    zi.X[j] = xi.X[j]

            # evaluate the new solution
            zi.F = self.problem.evaluate(zi)

            # update u probability
            if zi.F < xi.F:
                current_pop[i] = zi
            else:
                self.u_prob += p

        return current_pop

    def run_exploitation(self, current_pop):
        dim = len(current_pop[0].X)
        for i in range(len(current_pop)):
            xi = current_pop[i]
            if np.random.rand() < 0.5:
                # Ambush strategy
                if np.random.rand() < self.l_prob:
                    # Small jump towards 2 other pumas
                    random_puma = self.initialization.do(self.problem, 1, algorithm=self)[0]
                    random_puma_scaled = random_puma * 2 * np.random.rand() * np.exp(np.random.randn(1, dim))

                    zi = xi.copy()
                    zi.X = self.male_puma + random_puma_scaled - xi
                else:
                    # Long jump towards the best puma
                    factor = 2 * np.random.rand()
                    denominator = 2 * np.random.rand() - 1 + np.random.randn(1, dim)

                    r = 2 * np.random.rand() - 1
                    f1 = np.random.randn(1, dim) * np.exp(2 - (self.current_iter / self.n_max_iters))
                    w = np.random.randn(1, dim)
                    v = np.random.randn(1, dim)
                    f2 = w * (v**2) * np.cos(2 * np.random.rand() * w)

                    scaled_xi = f1 * r * xi
                    scaled_puma_male = f2 * (1 - r) * self.male_puma

                    zi = xi.copy()
                    zi.X = (factor * (scaled_xi + scaled_puma_male) / denominator) - self.male_puma
            else:
                # Run strategy
                mean_puma = np.mean(current_pop.X, axis=0) / len(current_pop)
                random_puma = self.initialization.do(self.problem, 1, algorithm=self)[0]
                beta = np.random.randint(0, 2)  # 0 or 1
                denominator = 1 + np.random.rand() * self.alpha  # scale

                zi = xi.copy()
                zi.X = (np.multiply(random_puma.X, mean_puma) - (1**beta) * xi) / denominator  # element wise product

            # repair the solution
            zi = self.repair(self.problem, zi)

            # evaluate the new solution
            zi.F = self.problem.evaluate(zi)

            if zi.F < xi.F:
                current_pop[i] = zi

        return current_pop
