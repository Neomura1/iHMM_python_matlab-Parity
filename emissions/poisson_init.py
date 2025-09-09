"""Initialise Poisson emission rate from a Gamma prior."""

from __future__ import annotations

import numpy as np


def poisson_init(Y, prior):
    """Sample ``lambda`` from the Gamma prior.

    ``Y`` is unused but kept for signature parity with the MATLAB code.
    """

    a0 = prior['a0']
    b0 = prior['b0']
    lam = np.random.gamma(a0, 1.0 / b0)
    return {'lam': lam}

