"""Simple RNG manager used to sync randomness with MATLAB implementations."""

from __future__ import annotations

import numpy as np


def rng_manager(seed: int = 1):
    """Seed the global NumPy RNG.

    The MATLAB code uses ``rng(seed,'twister')``; in NumPy we simply seed
    the default RNG which suffices for reproducibility in the examples and
    tests.
    """

    np.random.seed(int(seed))

