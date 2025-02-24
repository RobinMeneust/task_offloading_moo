from pymoo.core.sampling import Sampling


class TaskOffloadingSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return problem.dataset_generator.create_pop(n_samples)
