from ._univariate_selection import CoxPHSurvivalScorer, FastSurvivalSVMScorer
from .._cached import CachedFitMixin


class CachedCoxPHSurvivalScorer(CachedFitMixin, CoxPHSurvivalScorer):

    def __init__(self, memory, alpha=0, ties='breslow', n_iter=100):
        self.memory = memory
        super().__init__(alpha=alpha, ties=ties, n_iter=n_iter)


class CachedFastSurvivalSVMScorer(CachedFitMixin, FastSurvivalSVMScorer):

    def __init__(self, memory, alpha=1, rank_ratio=1.0, max_iter=20,
                 optimizer='avltree', random_state=777):
        self.memory = memory
        super().__init__(alpha=alpha, rank_ratio=rank_ratio, max_iter=max_iter,
                         optimizer=optimizer, random_state=random_state)
