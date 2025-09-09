"""Sample table counts ``m_{jk}`` for the Chinese restaurant franchise."""

from __future__ import annotations

import numpy as np


def crf_table_counts(counts, alpha, kappa):
    """Return matrix ``m`` with same shape as ``counts``.

    This implementation follows the standard Bernoulli sum construction
    ``m = sum_{i=1}^n Bernoulli(alpha/(alpha+i-1))`` and adds the sticky
    ``kappa`` mass to self-transition counts.
    """

    counts = np.asarray(counts, dtype=int)
    K = counts.shape[0]
    m = np.zeros_like(counts, dtype=int)
    for j in range(K):
        for k in range(K):
            n = counts[j, k]
            if n == 0:
                continue
            base = alpha
            if j == k:
                base += kappa
            for i in range(n):
                if np.random.rand() < base / (base + i):
                    m[j, k] += 1
    return m

