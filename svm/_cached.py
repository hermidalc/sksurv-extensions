from sksurv.svm import FastSurvivalSVM
from sklearn_extensions.cached import CachedFitMixin


class CachedFastSurvivalSVM(CachedFitMixin, FastSurvivalSVM):
    def __init__(
        self,
        memory,
        alpha=1,
        rank_ratio=1.0,
        fit_intercept=False,
        max_iter=20,
        verbose=False,
        tol=None,
        optimizer=None,
        random_state=None,
        timeit=False,
    ):
        super().__init__(
            alpha=alpha,
            rank_ratio=rank_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            verbose=verbose,
            tol=tol,
            optimizer=optimizer,
            random_state=random_state,
            timeit=timeit,
        )
        self.memory = memory
