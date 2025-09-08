"""Evaluate Poisson log likelihoods for all states."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def poisson_loglik(Y, theta_set):
    """Return ``K×T`` log-likelihood matrix for count observations ``Y``."""

    Y = np.asarray(Y)
    T, D = Y.shape
    K = len(theta_set)
    LL = np.empty((K, T))
    for k, th in enumerate(theta_set):
        lam = th['lam']
        ll = np.sum(Y * np.log(lam) - lam - gammaln(Y + 1), axis=1)
        LL[k, :] = ll
    return LL

