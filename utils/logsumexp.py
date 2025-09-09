"""Stable log-sum-exp utility.

This module mirrors the MATLAB helper ``logsumexp.m``.  The function
operates purely on ``numpy`` arrays and returns the log of the summed
exponentials along the requested axis while guarding against numerical
underflow/overflow.

Parameters
----------
A : array_like
    Input log probabilities.
axis : int, optional
    Axis over which to operate.  Default is ``0`` to match the MATLAB
    convention used throughout the code base.

Returns
-------
ndarray
    The log-sum-exp of ``A`` along ``axis``.
"""

from __future__ import annotations

import numpy as np


def logsumexp(A, axis: int = 0):
    """Compute ``log(sum(exp(A)))`` in a numerically stable way.

    The implementation follows the well known trick of subtracting the
    maximum before exponentiating.
    """

    A = np.asarray(A)
    if A.size == 0:
        return -np.inf

    # ``keepdims`` preserves dimensionality for broadcasting when
    # subtracting and later squeezing to MATLAB‑style behaviour.
    a_max = np.max(A, axis=axis, keepdims=True)
    # Guard against ``-inf`` which would propagate NaNs when exponentiated.
    a_max[~np.isfinite(a_max)] = 0.0
    lse = a_max + np.log(np.sum(np.exp(A - a_max), axis=axis, keepdims=True))
    return np.squeeze(lse, axis=axis)

