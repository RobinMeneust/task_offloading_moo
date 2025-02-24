from pymoo.core.algorithm import Algorithm


class PumaOptimizer(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup(self, problem, **kwargs):
        pass

    def _initialize_infill(self):
        pass

    def _initialize_advance(self, infills=None, **kwargs):
        pass

    def _infill(self):
        pass

    def _advance(self, infills=None, **kwargs):
        pass

    def _finalize(self):
        pass
