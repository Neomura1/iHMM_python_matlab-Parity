"""Draw a truncated GEM stick-breaking sequence."""

from __future__ import annotations

import numpy as np


def stick_breaking(gamma, K_trunc):
    v = np.random.beta(1.0, gamma, size=K_trunc)
    sticks = np.empty(K_trunc)
    remaining = 1.0
    for k in range(K_trunc - 1):
        sticks[k] = v[k] * remaining
        remaining *= 1.0 - v[k]
    sticks[-1] = remaining
    return sticks

