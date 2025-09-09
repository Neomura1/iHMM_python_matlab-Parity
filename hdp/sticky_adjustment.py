"""Apply sticky self-transition bias to count matrix."""

from __future__ import annotations

import numpy as np


def sticky_adjustment(kappa, counts):
    """Add ``kappa`` to the diagonal of the count matrix."""

    counts = np.asarray(counts, dtype=float)
    if kappa <= 0:
        return counts
    return counts + np.eye(counts.shape[0]) * kappa

