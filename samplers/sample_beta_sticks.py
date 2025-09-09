"""Sample the global stick weights ``beta`` from table counts."""

from __future__ import annotations

import numpy as np

from hdp.dirichlet_draw import dirichlet_draw


def sample_beta_sticks(counts, gamma, opt=None):
    m = counts.sum(axis=0)
    K = m.shape[0]
    alpha = m + gamma / K
    return dirichlet_draw(alpha)

