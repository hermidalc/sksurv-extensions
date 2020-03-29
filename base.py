"""
Base classes for all estimators.
"""


def is_survivalanalyzer(estimator):
    """Return True if the given estimator is (probably) a survivalanalyzer.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a survivalanalyzer and False otherwise.
    """

    return getattr(estimator, "_estimator_type", None) == "survivalanalyzer"
