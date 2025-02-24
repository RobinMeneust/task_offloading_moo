from pymoo.core.repair import Repair


class TaskOffloadingRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X = [problem.dataset_generator.repair_individual(x) for x in X]
        return X
