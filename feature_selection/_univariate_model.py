"""Univariate survival model feature selection."""

import numpy as np

from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn_extensions.feature_selection._base import ExtendedSelectorMixin
from sklearn_extensions.utils.validation import check_is_fitted, check_memory
from ..linear_model import FastCoxPHSurvivalAnalysis


def _get_scores(estimator, X, y, feature_idx_groups, **fit_params):
    scores = np.zeros(X.shape[1])
    if isinstance(estimator, CoxPHFitter):
        for js in feature_idx_groups:
            scores[js[0]] = (estimator.fit(X[:, js], y, **fit_params)
                             .log_likelihood_)
    else:
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
    estimator_ : an estimator
        The external estimator fit on the reduced dataset.

    scores_ : array-like of shape (n_features,)
        Feature scores.
    """

    def __init__(self, estimator, k='all', memory=None):
        self.estimator = estimator
        self.k = k
        self.memory = memory
        self._penalized_k = k

    def set_params(self, **params):
        super().set_params(**params)
        if 'k' in params:
            self._penalized_k = self.k
        return self

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
        penalty_factor = (
            feature_meta[estimator.penalty_factor_meta_col].to_numpy()
            if (feature_meta is not None
                and hasattr(estimator, 'penalty_factor_meta_col')
                and estimator.penalty_factor_meta_col is not None)
            else estimator.penalty_factor
            if (hasattr(estimator, 'penalty_factor')
                and estimator.penalty_factor is not None)
            else None)
        if penalty_factor is not None:
            unpenalized_feature_idxs = (np.where(penalty_factor == 0)[0]
                                        .tolist())
            if unpenalized_feature_idxs:
                penalized_feature_idxs = list(
                    set(feature_idxs) - set(unpenalized_feature_idxs))
                feature_idx_groups = [[j] + unpenalized_feature_idxs
                                      for j in penalized_feature_idxs]
                self.k = self._penalized_k + len(unpenalized_feature_idxs)
        if isinstance(estimator, CoxPHFitter):
            estimator.set_params(penalizer=1e-5)
        elif isinstance(estimator, CoxPHSurvivalAnalysis):
            estimator.set_params(alpha=1e-5)
        elif isinstance(estimator, FastCoxPHSurvivalAnalysis):
            estimator.set_params(alpha=0, penalty_factor=None)
        if hasattr(estimator, 'penalty_factor_meta_col'):
            estimator.set_params(penalty_factor_meta_col=None)
        scores = memory.cache(_get_scores)(estimator, X, y, feature_idx_groups,
                                           **fit_params)
        self.estimator_ = clone(self.estimator)
        if penalty_factor is not None and unpenalized_feature_idxs:
            for j in unpenalized_feature_idxs:
                scores[j] = 1.0
        self.scores_ = scores
        if isinstance(estimator, CoxPHSurvivalAnalysis):
            self.estimator_.set_params(
                alpha=penalty_factor[self.get_support()])
        elif isinstance(estimator, FastCoxPHSurvivalAnalysis):
            self.estimator_.set_params(
                penalty_factor=penalty_factor[self.get_support()])
        if hasattr(self.estimator_, 'penalty_factor_meta_col'):
            self.estimator_.set_params(penalty_factor_meta_col=None)
        self.estimator_.fit(self.transform(X), y, **fit_params)
        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X, **predict_params):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        **predict_params : dict of string -> object
            Parameters passed to the ``predict`` method of the estimator

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self)
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y, sample_weight=None):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the estimator.
        """
        check_is_fitted(self)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.estimator_.score(self.transform(X), y, **score_params)


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
