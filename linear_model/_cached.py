from sksurv.linear_model import CoxPHSurvivalAnalysis
from .._cached import CachedFitMixin


class CachedCoxPHSurvivalAnalysis(CachedFitMixin, CoxPHSurvivalAnalysis):

    def __init__(self, memory, alpha=0, ties='breslow', n_iter=100, tol=1e-9,
                 verbose=0):
        self.memory = memory
        super().__init__(alpha=alpha, ties=ties, n_iter=n_iter, tol=tol,
                         verbose=verbose)
