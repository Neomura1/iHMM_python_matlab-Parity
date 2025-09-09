"""Compute sufficient statistics for a diagonal Gaussian emission."""

from __future__ import annotations

import numpy as np


def gauss_diag_suffstats(Y, z, K=None):
    """Return counts and first/second order sums for each state.

    Parameters
    ----------
    Y : ndarray, shape (T, D)
        Observation sequence.
    z : ndarray, shape (T,)
        State sequence with values ``0..K-1``.
    K : int, optional
        Total number of states.  If ``None`` it is inferred from ``z``.
    """

    Y = np.asarray(Y)
    z = np.asarray(z, dtype=int)
    T, D = Y.shape
    if K is None:
        K = int(z.max()) + 1
    SS = {
        'n': np.zeros(K, dtype=int),
        'y_sum': np.zeros((K, D)),
        'y2_sum': np.zeros((K, D)),
    }
    for k in range(K):
        mask = z == k
        if np.any(mask):
            Yk = Y[mask]
            SS['n'][k] = Yk.shape[0]
            SS['y_sum'][k] = Yk.sum(axis=0)
            SS['y2_sum'][k] = (Yk ** 2).sum(axis=0)
    return SS

