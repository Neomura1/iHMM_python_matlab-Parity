"""Posterior update for a Normal–Inverse-Gamma prior (diagonal case)."""

from __future__ import annotations

import numpy as np


def gauss_diag_posterior(SS, prior):
    """Return posterior hyperparameters for each state."""

    K = len(SS['n'])
    post = []
    for k in range(K):
        n = SS['n'][k]
        y_sum = SS['y_sum'][k]
        y2_sum = SS['y2_sum'][k]
        mu0 = np.asarray(prior['mu0'])
        kappa0 = np.asarray(prior['kappa0'])
        a0 = np.asarray(prior['a0'])
        b0 = np.asarray(prior['b0'])

        kappaN = kappa0 + n
        muN = (kappa0 * mu0 + y_sum) / kappaN
        aN = a0 + 0.5 * n
        bN = b0 + 0.5 * (y2_sum + kappa0 * mu0**2 - kappaN * muN**2)

        post.append({'mu0': muN, 'kappa0': kappaN, 'a0': aN, 'b0': bN})
    return post

