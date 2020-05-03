"""Univariate survival model feature selection."""

import numpy as np

from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.utils import check_X_y
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn_extensions.feature_selection._base import ExtendedSelectorMixin
from sklearn_extensions.utils.validation import check_is_fitted, check_memory
from ..linear_model import FastCoxPHSurvivalAnalysis


def _get_scores(estimator, X, y, feature_idx_groups, **fit_params):
    scores = np.zeros(X.shape[1])
    for js in feature_idx_groups:
        Xjs = X[:, js]
        scores[js[0]] = estimator.fit(Xjs, y, **fit_params).score(Xjs, y)
    return scores


class SelectFromUnivariateSurvivalModel(ExtendedSelectorMixin,
                                        MetaEstimatorMixin, BaseEstimator):
    """Select features according to scores calculated from survival model
    fitting on each individual feature, including adjusting for any passed
    prognostic clinical or molecular features.

    Parameters
    ----------
    estimator : object
        The external estimator used to calculate univariate feature scores.

    k : int or "all" (default = "all")
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Feature scores.
    """

    def __init__(self, estimator, k='all', memory=None):
        self.estimator = estimator
        self.k = k
        self.memory = memory
        self._penalized_k = k

    def fit(self, X, y, feature_meta=None, **fit_params):
        """Fits an unpenalized model on each feature individually (including
        and adjusting for prognostic clinical or molecular features that are
        passed) and calculates each score. Then selects the ``k`` best scoring
        features and fits a final penalized model on these features (including
        and adjusting for unpenalized prognostic clinical or molecular features
        that are passed).

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The training input data matrix.

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator as the
            first field, and time of event or time of censoring as the second
            field.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        self._check_params(X, y, feature_meta)
        memory = check_memory(self.memory)
        feature_idxs = range(X.shape[1])
        feature_idx_groups = [[j] for j in feature_idxs]
        estimator = clone(self.estimator)
        if isinstance(estimator, CoxPHFitter):
            # for optimization numerical stability
            estimator.set_params(penalizer=1e-5)
        elif isinstance(estimator, CoxPHSurvivalAnalysis):
            # for optimization numerical stability
            estimator.set_params(alpha=1e-5)
        elif isinstance(estimator, FastCoxPHSurvivalAnalysis):
            penalty_factor = (
                feature_meta[estimator.penalty_factor_meta_col].to_numpy()
                if estimator.penalty_factor_meta_col is not None else
                estimator.penalty_factor
                if estimator.penalty_factor is not None else None)
            if penalty_factor is not None:
                unpenalized_feature_idxs = (
                    np.where(penalty_factor == 0)[0].tolist())
                if unpenalized_feature_idxs:
                    penalized_feature_idxs = list(
                        set(feature_idxs) - set(unpenalized_feature_idxs))
                    feature_idx_groups = [[j] + unpenalized_feature_idxs
                                          for j in penalized_feature_idxs]
                    self.k = self._penalized_k + len(unpenalized_feature_idxs)
            estimator.set_params(alpha=0, penalty_factor=None,
                                 penalty_factor_meta_col=None)
        scores = memory.cache(_get_scores)(estimator, X, y, feature_idx_groups,
                                           **fit_params)
        if (isinstance(estimator, FastCoxPHSurvivalAnalysis)
                and penalty_factor is not None and unpenalized_feature_idxs):
            for j in unpenalized_feature_idxs:
                scores[j] = 1.0
        self.scores_ = scores
        return self

    def _more_tags(self):
        estimator_tags = self.estimator._get_tags()
        return {'allow_nan': estimator_tags.get('allow_nan', True)}

    def _check_params(self, X, y, feature_meta):
        if not (self.k == 'all' or 0 <= self.k <= X.shape[1]):
            raise ValueError("k should be >=0, <= n_features = %d; got %r. "
                             "Use k='all' to return all features."
                             % (X.shape[1], self.k))
        if (hasattr(self.estimator, 'penalty_factor_meta_col')
                and self.estimator.penalty_factor_meta_col is not None):
            if feature_meta is None:
                raise ValueError('penalty_factor_meta_col specified but '
                                 'feature_meta not passed.')
            if (self.estimator.penalty_factor_meta_col not in
                    feature_meta.columns):
                raise ValueError('%s feature_meta column does not exist.'
                                 % self.estimator.penalty_factor_meta_col)

    def _get_support_mask(self):
        check_is_fitted(self)
        if self.k == 'all':
            return np.ones_like(self.scores_, dtype=bool)
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.k > 0:
            mask[np.argsort(self.scores_, kind='mergesort')[-self.k:]] = True
        return mask
