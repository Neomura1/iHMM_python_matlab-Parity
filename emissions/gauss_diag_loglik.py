"""Evaluate diagonal Gaussian log likelihoods for all states."""

from __future__ import annotations

import numpy as np


def gauss_diag_loglik(Y, theta_set):
    """Return a ``K×T`` array of log-likelihoods.

    Parameters
    ----------
    Y : ndarray, shape (T, D)
        Observation sequence.
    theta_set : list of dict
        Each element contains ``mu`` and ``sigma2`` for a state.
    """

    Y = np.asarray(Y)
    T, D = Y.shape
    K = len(theta_set)
    LL = np.empty((K, T))
    const = -0.5 * D * np.log(2 * np.pi)
    for k, th in enumerate(theta_set):
        mu = th['mu']
        var = th['sigma2']
        diff = Y - mu
        ll = const - 0.5 * np.sum(np.log(var)) - 0.5 * np.sum(diff**2 / var, axis=1)
        LL[k, :] = ll
    return LL

