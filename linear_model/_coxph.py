from sksurv.linear_model import CoxnetSurvivalAnalysis


class FastCoxPHSurvivalAnalysis(CoxnetSurvivalAnalysis):
    """Fast Cox proportional hazards model using Coxnet with settings to
    reproduce standard Cox L2 ridge regression objective function and
    supporting coefficient penalty factors.

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

    max_iter : int (default=500000)
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

    def __init__(self, alpha=0, l1_ratio=1e-30, penalty_factor=None,
                 normalize=False, copy_X=True, tol=1e-15, max_iter=500000,
                 verbose=False, fit_baseline_model=False):
        super().__init__(
            l1_ratio=l1_ratio, penalty_factor=penalty_factor,
            normalize=normalize, copy_X=copy_X, tol=tol, max_iter=max_iter,
            verbose=verbose, fit_baseline_model=fit_baseline_model)
        self.alpha = alpha

    def fit(self, X, y):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator as the
            first field, and time of event or time of censoring as the second
            field.

        Returns
        -------
        self
        """
        self.alphas = [self.alpha/X.shape[0]]
        super().fit(X, y)
