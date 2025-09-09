"""Sample from a Dirichlet distribution."""

from __future__ import annotations

import numpy as np


def dirichlet_draw(alpha_vec):
    alpha_vec = np.asarray(alpha_vec, dtype=float)
    if np.any(alpha_vec <= 0):
        raise ValueError('alpha parameters must be positive')
    sample = np.random.gamma(alpha_vec, 1.0)
    return sample / sample.sum()

