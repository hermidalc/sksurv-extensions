"""CoxPH extensions."""

import numbers
import numpy as np

from sklearn.utils import check_X_y
from sksurv.linear_model import CoxPHSurvivalAnalysis


class ExtendedCoxPHSurvivalAnalysis(CoxPHSurvivalAnalysis):
    """Cox proportional hazards model.

    There are two possible choices for handling tied event times.
    The default is Breslow's method, which considers each of the
    events at a given time as distinct. Efron's method is more
    accurate if there are a large number of ties. When the number
    of ties is small, the estimated coefficients by Breslow's and
    Efron's method are quite close. Uses Newton-Raphson optimization.

    See [1]_, [2]_, [3]_ for further description.

    Parameters
    ----------
    alpha : float, ndarray of shape (n_features,), optional, default: 0
        Regularization parameter for ridge regression penalty.
        If a single float, the same penalty is used for all features.
        If an array, there must be one penalty for each feature.
        If you want to include a subset of features without penalization,
        set the corresponding entries to 0.

    ties : "breslow" | "efron", optional, default: "breslow"
        The method to handle tied event times. If there are
        no tied event times all the methods are equivalent.

    n_iter : int, optional, default: 100
        Maximum number of iterations.

    tol : float, optional, default: 1e-9
        Convergence criteria. Convergence is based on the negative log-likelihood::

        |1 - (new neg. log-likelihood / old neg. log-likelihood) | < tol

    verbose : int, optional, default: 0
        Specified the amount of additional debug information
        during optimization.

    base_alpha : float (default=1e-5)
        Base regularization parameter to support regression convergence.

    penalty_factor_meta_col : str (default=None)
        Feature metadata column name to use for penalty factors.  These will be
        used to adjust ``alpha``.  This is ignored if ``alpha`` is an array
        of penalties.

    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Coefficients of the model

    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Estimated baseline cumulative hazard function.

    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Estimated baseline survival function.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    See also
    --------
    sksurv.linear_model.CoxnetSurvivalAnalysis
        Cox proportional hazards model with l1 (LASSO) and l2 (ridge) penalty.

    References
    ----------
    .. [1] Cox, D. R. Regression models and life tables (with discussion).
           Journal of the Royal Statistical Society. Series B, 34, 187-220, 1972.
    .. [2] Breslow, N. E. Covariance Analysis of Censored Survival Data.
           Biometrics 30 (1974): 89–99.
    .. [3] Efron, B. The Efficiency of Cox’s Likelihood Function for Censored Data.
           Journal of the American Statistical Association 72 (1977): 557–565.
    """

    def __init__(
        self,
        alpha=0,
        ties="efron",
        n_iter=1000,
        tol=1e-9,
        verbose=0,
        base_alpha=1e-5,
        penalty_factor_meta_col=None,
    ):
        super().__init__(
            alpha=alpha, ties=ties, n_iter=n_iter, tol=tol, verbose=verbose
        )
        self.base_alpha = base_alpha
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
        if (
            isinstance(self.alpha, (numbers.Real, numbers.Integral))
            and self.penalty_factor_meta_col is not None
        ):
            alphas = np.full(X.shape[1], self.alpha, dtype=float)
            penalty_factor = feature_meta[self.penalty_factor_meta_col].to_numpy(
                dtype=float
            )
            if alphas.shape[0] != penalty_factor.shape[0]:
                raise ValueError(
                    "Length of alphas ({}) must match length of "
                    "penalty_factor_meta_col ({}).".format(
                        alphas.shape[0], penalty_factor.shape[0]
                    )
                )
            alphas = alphas * penalty_factor
            alphas[alphas < self.base_alpha] = self.base_alpha
            self.alpha = alphas
        return super().fit(X, y)

    def predict(self, X, feature_meta=None):
        """Predict risk scores.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test data of which to calculate log-likelihood from

        feature_meta : Ignored.

        Returns
        -------
        risk_score : array, shape = (n_samples,)
            Predicted risk scores.
        """
        return super().predict(X)

    # double underscore to not override parent
    def __check_params(self, X, y, feature_meta):
        if self.penalty_factor_meta_col is not None:
            if feature_meta is None:
                raise ValueError(
                    "penalty_factor_meta_col specified but " "feature_meta not passed."
                )
            if self.penalty_factor_meta_col not in feature_meta.columns:
                raise ValueError(
                    "%s feature_meta column does not exist."
                    % self.penalty_factor_meta_col
                )
            if X.shape[1] != feature_meta.shape[0]:
                raise ValueError(
                    "X ({:d}) and feature_meta ({:d}) have "
                    "different feature dimensions".format(
                        X.shape[1], feature_meta.shape[0]
                    )
                )
