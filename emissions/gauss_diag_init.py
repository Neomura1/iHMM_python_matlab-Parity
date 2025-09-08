"""Initialise diagonal Gaussian emission parameters from the prior."""

from __future__ import annotations

import numpy as np


def gauss_diag_init(Y, prior):
    """Sample ``mu`` and ``sigma2`` from a Normal–Inverse-Gamma prior."""

    D = Y.shape[1]
    mu0 = np.asarray(prior['mu0'])
    kappa0 = np.asarray(prior['kappa0'])
    a0 = np.asarray(prior['a0'])
    b0 = np.asarray(prior['b0'])

    sigma2 = 1.0 / np.random.gamma(a0, 1.0 / b0, size=D)
    mu = mu0 + np.random.randn(D) * np.sqrt(sigma2 / kappa0)
    return {'mu': mu, 'sigma2': sigma2}

