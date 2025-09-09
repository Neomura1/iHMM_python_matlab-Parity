"""Draw a single sample from a discrete distribution.

The function expects a one dimensional array ``p`` containing (possibly
unnormalised) probabilities.  The return value is the sampled index in
``0 … len(p)-1``.  This mirrors the MATLAB helper ``sample_discrete.m``
used throughout the original code base.
"""

from __future__ import annotations

import numpy as np


def sample_discrete(p):
    """Return an index sampled according to ``p``."""

    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p must be a 1D array")
    total = p.sum()
    if total <= 0:
        raise ValueError("distribution has non-positive mass")
    cdf = np.cumsum(p / total)
    r = np.random.rand()
    return int(np.searchsorted(cdf, r))

