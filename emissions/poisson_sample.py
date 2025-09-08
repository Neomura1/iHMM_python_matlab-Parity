"""Sample Poisson rate parameters from the posterior."""

from __future__ import annotations

import numpy as np


def poisson_sample(post):
    a0 = post['a0']
    b0 = post['b0']
    lam = np.random.gamma(a0, 1.0 / b0)
    return {'lam': lam}

