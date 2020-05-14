from ._coxph import ExtendedCoxPHSurvivalAnalysis
from .._cached import CachedFitMixin


class CachedExtendedCoxPHSurvivalAnalysis(CachedFitMixin,
                                          ExtendedCoxPHSurvivalAnalysis):

    def __init__(self, memory, alpha=0, ties='efron', n_iter=1000, tol=1e-9,
                 verbose=0, penalty_factor_meta_col=None):
        super().__init__(alpha=alpha, ties=ties, n_iter=n_iter, tol=tol,
                         verbose=verbose,
                         penalty_factor_meta_col=penalty_factor_meta_col)
        self.memory = memory
