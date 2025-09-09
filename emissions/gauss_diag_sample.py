"""Sample emission parameters from a diagonal Gaussian posterior."""

from __future__ import annotations

import numpy as np


def gauss_diag_sample(post):
    """Draw ``mu`` and ``sigma2`` from the posterior hyperparameters."""

    mu0 = np.asarray(post['mu0'])
    kappa0 = np.asarray(post['kappa0'])
    a0 = np.asarray(post['a0'])
    b0 = np.asarray(post['b0'])

    sigma2 = 1.0 / np.random.gamma(a0, 1.0 / b0)
    mu = mu0 + np.random.randn(*mu0.shape) * np.sqrt(sigma2 / kappa0)
    return {'mu': mu, 'sigma2': sigma2}

