"""Compute transition counts ``n_{jk}`` from state sequences."""

from __future__ import annotations

import numpy as np


def crf_customer_counts(z_cell):
    """Return ``K×K`` matrix of transition counts."""

    if not z_cell:
        return np.zeros((0, 0), dtype=int)
    K = int(max(z.max() for z in z_cell)) + 1
    counts = np.zeros((K, K), dtype=int)
    for z in z_cell:
        for t in range(1, len(z)):
            counts[z[t - 1], z[t]] += 1
    return counts

