# Authors: Leandro Hermida <hermidal@cs.umd.edu>
#
# License: BSD 3 clause

from sklearn.model_selection import StratifiedKFold


class SurvivalStratifiedKFold(StratifiedKFold):
    """Survival Stratified K-Folds cross-validator

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds for a survival structured array. The folds are made by
    preserving the percentage of samples for each survival status.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle each status's samples before splitting into batches.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``shuffle`` is True. This should be left
        to None if ``shuffle`` is False.

    Notes
    -----
    The implementation is designed to:

    * Generate test sets such that all contain the same distribution of
      ssurvival status, or as close as possible.
    * Be invariant to status label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from status k in some test set were
      contiguous in y, or separated in y by samples from statuses other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.
    """

    def _make_test_folds(self, X, y):
        # make sksurv structured array look like binary class (on status)
        y = y[y.dtype.names[0]].astype(int)
        test_folds = super()._make_test_folds(X, y)
        return test_folds
