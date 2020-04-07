"""Coxnet extensions."""

from sklearn.utils import check_X_y
from sksurv.linear_model import CoxnetSurvivalAnalysis


class ExtendedCoxnetSurvivalAnalysis(CoxnetSurvivalAnalysis):
    """Cox proportional hazards model with elastic net penalty and support for
    setting feature penalty_factor in a Pipeline context.

    See [1]_ for further description.

    Parameters
    ----------
    n_alphas : int (default=100)
        Number of alphas along the regularization path.

    alphas : array-like or None (default=None)
        List of alphas where to compute the models. If ``None`` alphas are set
        automatically.

    alpha_min_ratio : float (default=0.0001)
        Determines minimum alpha of the regularization path if ``alphas`` is
        ``None``. The smallest value for alpha is computed as the fraction of
        the data derived maximum alpha (i.e. the smallest value for which all
        coefficients are zero).

    l1_ratio : float (default=0.5)
        The ElasticNet mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty.
        For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    penalty_factor : array-like or None (default=None)
        Separate penalty factors can be applied to each coefficient. This is a
        number that multiplies alpha to allow differential shrinkage. Can be 0
        for some variables, which implies no shrinkage, and that variable is
        always included in the model. Default is 1 for all variables. Note: the
        penalty factors are internally rescaled to sum to n_features, and the
        alphas sequence will reflect this change.

    penalty_factor_meta_col : str (default=None)
        Feature metadata column name to use for ``penalty_factor``. This
        overrides any ``penalty_factor`` setting if both are set.

    normalize : boolean (default=False)
        If True, the features X will be normalized before optimization by
        subtracting the mean and dividing by the l2-norm. If you wish to
        standardize, please use :class:`sklearn.preprocessing.StandardScaler`
        before calling ``fit`` on an estimator with ``normalize=False``.

    copy_X : boolean (default=True)
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float (default=1e-7)
        The tolerance for the optimization: optimization continues until all
        updates are smaller than ``tol``.

    max_iter : int (default=100000)
        The maximum number of iterations.

    verbose : bool (default=False)
        Whether to print additional information during optimization.

    fit_baseline_model : bool (default=False)
        Whether to estimate baseline survival function and baseline cumulative
        hazard function for each alpha. If enabled,
        :meth:`predict_cumulative_hazard_function` and
        :meth:`predict_survival_function` can be used to obtain predicted
        cumulative hazard function and survival function.

    Attributes
    ----------
    alphas_ : ndarray, shape=(n_alphas,)
        The actual sequence of alpha values used.

    penalty_factor_ : ndarray, shape=(n_features,)
        The actual penalty factors used.

    coef_ : ndarray, shape=(n_features, n_alphas)
        Matrix of coefficients.

    deviance_ratio_ : ndarray, shape=(n_alphas,)
        The fraction of (null) deviance explained.

    References
    ----------
    .. [1] Simon N, Friedman J, Hastie T, Tibshirani R.
           Regularization paths for Cox’s proportional hazards model via \
           coordinate descent.
           Journal of statistical software. 2011 Mar;39(5):1.
    """

    def __init__(self, n_alphas=100, alphas=None, alpha_min_ratio=0.0001,
                 l1_ratio=0.5, penalty_factor=None,
                 penalty_factor_meta_col=None, normalize=False, copy_X=True,
                 tol=1e-7, max_iter=100000, verbose=False,
                 fit_baseline_model=False):
        super().__init__(
            n_alphas=n_alphas, alphas=alphas, alpha_min_ratio=alpha_min_ratio,
            l1_ratio=l1_ratio, penalty_factor=penalty_factor,
            normalize=normalize, copy_X=copy_X, tol=tol, max_iter=max_iter,
            verbose=verbose, fit_baseline_model=fit_baseline_model)
        self.penalty_factor_meta_col = penalty_factor_meta_col

    def fit(self, X, y, feature_meta=None):
        """Fits the estimator.

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

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        self.__check_params(X, y, feature_meta)
        if self.penalty_factor_meta_col is not None:
            self.penalty_factor = (feature_meta[self.penalty_factor_meta_col]
                                   .to_numpy())
        return super().fit(X, y)

    def predict(self, X, alpha=None, feature_meta=None):
        """The linear predictor of the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test data of which to calculate log-likelihood from

        alpha : float (default=None)
            Constant that multiplies the penalty terms. If the same alpha was
            used during training, exact coefficients are used, otherwise
            coefficients are interpolated from the closest alpha values that
            were used during training. If set to ``None``, the last alpha in
            the solution path is used.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        T : array, shape = (n_samples,)
            The predicted decision function
        """
        return super().predict(X, alpha)

    # double underscore to not override parent
    def __check_params(self, X, y, feature_meta):
        if self.penalty_factor_meta_col is not None:
            if feature_meta is None:
                raise ValueError('penalty_factor_meta_col specified but '
                                 'feature_meta not passed.')
            if self.penalty_factor_meta_col not in feature_meta.columns:
                raise ValueError('%s feature_meta column does not exist.'
                                 % self.penalty_factor_meta_col)


class FastCoxPHSurvivalAnalysis(ExtendedCoxnetSurvivalAnalysis):
    """Fast Cox proportional hazards model using Coxnet with settings to
    reproduce standard Cox L2 ridge regression objective function and support
    for setting feature penalty_factor in a Pipeline context.

    See [1]_ for further description.

    Parameters
    ----------
    alpha : float (default=0)
        Regularization parameter for ridge regression penalty.

    l1_ratio : float (default=1e-30)
        The ElasticNet mixing parameter to reproduce L2 ridge regression.
        **Do not change unless there are issues with a particular dataset.**

    penalty_factor : array-like or None (default=None)
        Separate penalty factors can be applied to each coefficient. This is a
        number that multiplies alpha to allow differential shrinkage. Can be 0
        for some variables, which implies no shrinkage, and that variable is
        always included in the model. Default is 1 for all variables. Note: the
        penalty factors are internally rescaled to sum to n_features, and the
        alphas sequence will reflect this change.

    penalty_factor_meta_col : str (default=None)
        Feature metadata column name to use for ``penalty_factor``. This
        overrides any ``penalty_factor`` setting if both are set.

    normalize : boolean (default=False)
        If True, the features X will be normalized before optimization by
        subtracting the mean and dividing by the l2-norm. If you wish to
        standardize, please use :class:`sklearn.preprocessing.StandardScaler`
        before calling ``fit`` on an estimator with ``normalize=False``.

    copy_X : boolean (default=True)
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float (default=1e-15)
        The tolerance for the optimization: optimization continues until all
        updates are smaller than ``tol``. **Do not change unless there are
        issues with a particular dataset.**

    max_iter : int (default=1000000)
        The maximum number of iterations.

    verbose : bool (default=False)
        Whether to print additional information during optimization.

    fit_baseline_model : bool (default=False)
        Whether to estimate baseline survival function and baseline cumulative
        hazard function for each alpha. If enabled,
        :meth:`predict_cumulative_hazard_function` and
        :meth:`predict_survival_function` can be used to obtain predicted
        cumulative hazard function and survival function.

    Attributes
    ----------
    alphas_ : ndarray, shape=(1,)
        The actual sequence of alpha values used.

    penalty_factor_ : ndarray, shape=(n_features,)
        The actual penalty factors used.

    coef_ : ndarray, shape=(n_features, 1)
        Matrix of coefficients.

    deviance_ratio_ : ndarray, shape=(1,)
        The fraction of (null) deviance explained.

    References
    ----------
    .. [1] Simon N, Friedman J, Hastie T, Tibshirani R.
           Regularization paths for Cox’s proportional hazards model via \
           coordinate descent.
           Journal of statistical software. 2011 Mar;39(5):1.
    """

    def __init__(self, alpha=0, l1_ratio=1e-40, penalty_factor=None,
                 penalty_factor_meta_col=None, normalize=False, copy_X=True,
                 tol=1e-16, max_iter=1000000, verbose=False,
                 fit_baseline_model=False):
        super().__init__(
            l1_ratio=l1_ratio, penalty_factor=penalty_factor,
            penalty_factor_meta_col=penalty_factor_meta_col,
            normalize=normalize, copy_X=copy_X, tol=tol, max_iter=max_iter,
            verbose=verbose, fit_baseline_model=fit_baseline_model)
        self.alpha = alpha

    def fit(self, X, y, feature_meta=None):
        """Fits the estimator.

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

        Returns
        -------
        self : object
        """
        self.alphas = [self.alpha / X.shape[0]]
        return super().fit(X, y, feature_meta)
