"""Sufficient statistics for Poisson emissions."""

from __future__ import annotations

import numpy as np


def poisson_suffstats(Y, z, K=None):
    Y = np.asarray(Y)
    z = np.asarray(z, dtype=int)
    if K is None:
        K = int(z.max()) + 1
    SS = {
        'n': np.zeros(K, dtype=int),
        'y_sum': np.zeros(K),
    }
    for k in range(K):
        mask = z == k
        if np.any(mask):
            Yk = Y[mask]
            SS['n'][k] = Yk.shape[0]
            SS['y_sum'][k] = Yk.sum()
    return SS

