"""Normalise log-probabilities along a given axis.

The routine is used extensively when working with probabilities in the
log-domain.  Given an array ``logP`` it subtracts the log-sum-exp so that
``exp(logP)`` sums to one along ``axis``.
"""

from __future__ import annotations

import numpy as np

from .logsumexp import logsumexp


def normalize_log(logP, axis: int = 0):
    """Normalise ``logP`` along ``axis`` and return the log normaliser."""

    logP = np.asarray(logP)
    lse = logsumexp(logP, axis=axis)
    # ``np.expand_dims`` to broadcast subtraction similar to MATLAB bsxfun
    logP_norm = logP - np.expand_dims(lse, axis=axis)
    return logP_norm, lse

