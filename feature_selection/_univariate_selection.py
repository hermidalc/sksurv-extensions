"""Survival univariate feature selection."""

import numpy as np

from sklearn.base import BaseEstimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM


######################################################################
# Base scorer class

class BaseSurvivalScorer(BaseEstimator):
    """Base survival univariate feature scorer."""


######################################################################
# Specific scorer classes

class CoxPHSurvivalScorer(BaseSurvivalScorer):
    """Compute CoxPHSurvivalAnalysis feature C-index scores.

    Parameters
    ----------
    alpha : float (default = 0)
        Regularization parameter for ridge regression penalty.

    ties : string (default = "breslow")
        The method to handle tied event times. If there are no tied event times
        all the methods are equivalent.

    n_iter : int (default = 100)
        Maximum number of iterations.

    Attributes
    ----------
    scores_ : array, shape = (n_features,)
        Feature C-index scores.
    """

    def __init__(self, alpha=0, ties='breslow', n_iter=100):
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter

    def fit(self, X, y):
        """Run feature scorer on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.

        y : array_like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        m = CoxPHSurvivalAnalysis(alpha=self.alpha, ties=self.ties,
                                  n_iter=self.n_iter)
        scores = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            Xj = X[:, [j]]
            scores[j] = m.fit(Xj, y).score(Xj, y)
        self.scores_ = scores
        return self


class FastSurvivalSVMScorer(BaseSurvivalScorer):
    """Compute FastSurvivalSVM feature C-index scores.

    Parameters
    ----------
    alpha : float (default = 1)
        Weight of penalizing the squared hinge loss in the objective function.

    rank_ratio : float (default = 1.0)
        Mixing parameter between regression and ranking objective with
        0 <= rank_ratio <= 1. If rank_ratio = 1, only ranking is performed, if
        rank_ratio = 0, only regression is performed. A non-zero value is only
        allowed if optimizer is one of "avltree", "rbtree", or "direct-count".

    max_iter : int (default = 20)
        Maximum number of iterations to perform in Newton optimization.

    optimizer : string (default = "avltree")
        Optimizer to use.

    random_state : int (default = 777)
        Random seed.

    Attributes
    ----------
    scores_ : array, shape = (n_features,)
        Feature C-index scores.
    """

    def __init__(self, alpha=1, rank_ratio=1.0, max_iter=20,
                 optimizer='avltree', random_state=777):
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.random_state = random_state

    def fit(self, X, y):
        """Run feature scorer on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.

        y : array_like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        m = FastSurvivalSVM(alpha=self.alpha, rank_ratio=self.rank_ratio,
                            max_iter=self.max_iter, optimizer=self.optimizer,
                            random_state=self.random_state)
        scores = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            Xj = X[:, [j]]
            scores[j] = m.fit(Xj, y).score(Xj, y)
        self.scores_ = scores
        return self
