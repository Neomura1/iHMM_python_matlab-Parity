"""Sample transition probability rows given counts and global weights."""

from __future__ import annotations

import numpy as np


def resample_row_transitions(counts, alpha, kappa, beta, opt=None):
    counts = np.asarray(counts, dtype=float)
    K = counts.shape[0]
    Pi = np.zeros_like(counts)
    base = alpha * beta
    for j in range(K):
        alpha_row = base.copy()
        if kappa > 0:
            alpha_row[j] += kappa
        Pi[j, :] = np.random.dirichlet(alpha_row + counts[j, :])
    return Pi

